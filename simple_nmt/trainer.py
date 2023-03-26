import numpy as np

import torch
from torch import optim
import torch.nn.utils as torch_utils
from torch.cuda.amp import autocast
from torch.cuda.amp import GradScaler

from ignite.engine import Engine
from ignite.engine import Events
from ignite.metrics import RunningAverage
from ignite.contrib.handlers.tqdm_logger import ProgressBar

from simple_nmt.utils import get_grad_norm, get_parameter_norm


VERBOSE_SILENT = 0
VERBOSE_EPOCH_WISE = 1
VERBOSE_BATCH_WISE = 2


class MaximumLikelihoodEstimationEngine(Engine):

    def __init__(self, func, model, crit, optimizer, lr_scheduler, config):
        self.model = model
        self.crit = crit
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler
        self.config = config

        super().__init__(func)

        self.best_loss = np.inf
        self.scaler = GradScaler()

    @staticmethod 
    # @staticmethod 데코레이터를 메서드 위에 붙이면 해당 메서드는 인스턴스의 속성에 접근하지 않기 때문에 첫 번째 인자로 self가 필요하지 않습니다. 
    # 그리고 해당 메서드는 클래스 이름을 통해 호출될 수 있습니다.
    #@profile
    def train(engine, mini_batch):
        # You have to reset the gradients of all model parameters
        # before to take another step in gradient descent.
        engine.model.train()
        # 업데이트당 iteration 숫자만큼의 한번씩씩
        if engine.state.iteration % engine.config.iteration_per_update == 1 or \
            engine.config.iteration_per_update == 1: # 1이라면 매 iteration마다 업데이트 한다는 것
            if engine.state.iteration > 1: 
                engine.optimizer.zero_grad() #zero_grad를 불러준다.

        device = next(engine.model.parameters()).device
        mini_batch.src = (mini_batch.src[0].to(device), mini_batch.src[1])
        mini_batch.tgt = (mini_batch.tgt[0].to(device), mini_batch.tgt[1])

        # Raw target variable has both BOS and EOS token. 
        # The output of sequence-to-sequence does not have BOS token. 
        # Thus, remove BOS token for reference.
        # src는 그대로 해당 텐서와 length를 넣어야 한다.,
        # decoder target y는 length빼고 해당문장의 텐서만 들어간다.
        # 모델에 넣을때는 bos + 기타/ 정답 나온거 비교하는 loss구하기 위한것은 bos 빼고 기타+ eos
        # [:, 1:] 미니 배치는 다가져옴, 그리고 첫번째 즉, bos빼고 다 가져옴
        x, y = mini_batch.src, mini_batch.tgt[0][:, 1:] 
        # |x| = (batch_size, length)
        # |y| = (batch_size, length)

        with autocast(not engine.config.off_autocast):
            # Take feed-forward
            # Similar as before, the input of decoder does not have EOS token.
            # Thus, remove EOS token for decoder input.
            y_hat = engine.model(x, mini_batch.tgt[0][:, :-1])
            # |y_hat| = (batch_size, length, output_size)

            loss = engine.crit(
                #
                y_hat.contiguous().view(-1, y_hat.size(-1)), # -1 부분이 batch_size * length가 될것이고 쭉 펴준다.
                y.contiguous().view(-1) # y도 batch_size * length로 쭉 펴준다.
            )
            # loss 를 구하면 minibatch size로 나눠준다., 나중에 ga 수행할것이기 때문에 이걸로 나눠줘 평균을 해준다.
            # 화면에는 loss 그리고 실제 역전파는 아래코드를 실행할 것이다. 
            backward_target = loss.div(y.size(0)).div(engine.config.iteration_per_update)

        if engine.config.gpu_id >= 0 and not engine.config.off_autocast:
            engine.scaler.scale(backward_target).backward()
        else:
            backward_target.backward()

        word_count = int(mini_batch.tgt[1].sum())
        p_norm = float(get_parameter_norm(engine.model.parameters()))
        g_norm = float(get_grad_norm(engine.model.parameters()))

        if engine.state.iteration % engine.config.iteration_per_update == 0 and \
            engine.state.iteration > 0:
            # In orther to avoid gradient exploding, we apply gradient clipping.
            torch_utils.clip_grad_norm_(
                engine.model.parameters(),
                engine.config.max_grad_norm,
            )
            # Take a step of gradient descent. # cpu를 썼을때 스케일러를 거치지 않게 한ㄷ.
            if engine.config.gpu_id >= 0 and not engine.config.off_autocast:
                # Use scaler instead of engine.optimizer.step() if using GPU.
                engine.scaler.step(engine.optimizer)
                engine.scaler.update()
            else:
                engine.optimizer.step()

        loss = float(loss / word_count)
        ppl = np.exp(loss)

        return {
            'loss': loss,
            'ppl': ppl,
            '|param|': p_norm if not np.isnan(p_norm) and not np.isinf(p_norm) else 0.,
            '|g_param|': g_norm if not np.isnan(g_norm) and not np.isinf(g_norm) else 0.,
        } # none나 inf가 뜨면 평균을 할 수 없기 때문에 0을 리턴한다.

    @staticmethod
    def validate(engine, mini_batch):
        engine.model.eval()

        with torch.no_grad():
            device = next(engine.model.parameters()).device
            mini_batch.src = (mini_batch.src[0].to(device), mini_batch.src[1])
            mini_batch.tgt = (mini_batch.tgt[0].to(device), mini_batch.tgt[1])

            x, y = mini_batch.src, mini_batch.tgt[0][:, 1:]
            # |x| = (batch_size, length)
            # |y| = (batch_size, length)

            with autocast(not engine.config.off_autocast):
                y_hat = engine.model(x, mini_batch.tgt[0][:, :-1])
                # |y_hat| = (batch_size, n_classes)
                loss = engine.crit(
                    y_hat.contiguous().view(-1, y_hat.size(-1)),
                    y.contiguous().view(-1),
                )
        
        word_count = int(mini_batch.tgt[1].sum())
        loss = float(loss / word_count)
        ppl = np.exp(loss)

        return {
            'loss': loss,
            'ppl': ppl,
        }

    @staticmethod
    def attach(
        train_engine, validation_engine,
        training_metric_names = ['loss', 'ppl', '|param|', '|g_param|'],
        validation_metric_names = ['loss', 'ppl'],
        verbose=VERBOSE_BATCH_WISE,
    ):
        # Attaching would be repaeted for serveral metrics.
        # Thus, we can reduce the repeated codes by using this function.
        def attach_running_average(engine, metric_name):
            RunningAverage(output_transform=lambda x: x[metric_name]).attach(
                engine,
                metric_name,
            )

        for metric_name in training_metric_names:
            attach_running_average(train_engine, metric_name)

        if verbose >= VERBOSE_BATCH_WISE:
            pbar = ProgressBar(bar_format=None, ncols=120)
            pbar.attach(train_engine, training_metric_names)

        if verbose >= VERBOSE_EPOCH_WISE:
            @train_engine.on(Events.EPOCH_COMPLETED)
            def print_train_logs(engine):
                avg_p_norm = engine.state.metrics['|param|']
                avg_g_norm = engine.state.metrics['|g_param|']
                avg_loss = engine.state.metrics['loss']

                print('Epoch {} - |param|={:.2e} |g_param|={:.2e} loss={:.4e} ppl={:.2f}'.format(
                    engine.state.epoch,
                    avg_p_norm,
                    avg_g_norm,
                    avg_loss,
                    np.exp(avg_loss),
                ))

        for metric_name in validation_metric_names:
            attach_running_average(validation_engine, metric_name)

        if verbose >= VERBOSE_BATCH_WISE:
            pbar = ProgressBar(bar_format=None, ncols=120)
            pbar.attach(validation_engine, validation_metric_names)

        if verbose >= VERBOSE_EPOCH_WISE:
            @validation_engine.on(Events.EPOCH_COMPLETED)
            def print_valid_logs(engine):
                avg_loss = engine.state.metrics['loss']

                print('Validation - loss={:.4e} ppl={:.2f} best_loss={:.4e} best_ppl={:.2f}'.format(
                    avg_loss,
                    np.exp(avg_loss),
                    engine.best_loss,
                    np.exp(engine.best_loss),
                ))

    @staticmethod
    def resume_training(engine, resume_epoch):
        engine.state.iteration = (resume_epoch - 1) * len(engine.state.dataloader)
        engine.state.epoch = (resume_epoch - 1)

    @staticmethod
    def check_best(engine):
        loss = float(engine.state.metrics['loss'])
        if loss <= engine.best_loss:
            engine.best_loss = loss

    @staticmethod
    def save_model(engine, train_engine, config, src_vocab, tgt_vocab):
        avg_train_loss = train_engine.state.metrics['loss']
        avg_valid_loss = engine.state.metrics['loss']

        # Set a filename for model of last epoch.
        # We need to put every information to filename, as much as possible.
        model_fn = config.model_fn.split('.')
        
        model_fn = model_fn[:-1] + ['%02d' % train_engine.state.epoch,
                                    '%.2f-%.2f' % (avg_train_loss,
                                                   np.exp(avg_train_loss)
                                                   ),
                                    '%.2f-%.2f' % (avg_valid_loss,
                                                   np.exp(avg_valid_loss)
                                                   )
                                    ] + [model_fn[-1]]

        model_fn = '.'.join(model_fn)

        # Unlike other tasks, we need to save current model, not best model.
        torch.save(
            {
                'model': engine.model.state_dict(),
                'opt': train_engine.optimizer.state_dict(),
                'config': config,
                'src_vocab': src_vocab,
                'tgt_vocab': tgt_vocab,
            }, model_fn
        )
    

class SingleTrainer():

    def __init__(self, target_engine_class, config):
        self.target_engine_class = target_engine_class
        self.config = config

    def train(
        self,
        model, crit, optimizer,
        train_loader, valid_loader,
        src_vocab, tgt_vocab,
        n_epochs,
        lr_scheduler=None
    ):
        # Declare train and validation engine with necessary objects.
        train_engine = self.target_engine_class(
            self.target_engine_class.train,
            model,
            crit,
            optimizer,
            lr_scheduler,
            self.config
        )
        validation_engine = self.target_engine_class(
            self.target_engine_class.validate,
            model,
            crit,
            optimizer=None,
            lr_scheduler=None,
            config=self.config
        )

        # Do necessary attach procedure to train & validation engine.
        # Progress bar and metric would be attached.
        self.target_engine_class.attach(
            train_engine,
            validation_engine,
            verbose=self.config.verbose
        )

        # After every train epoch, run 1 validation epoch.
        # Also, apply LR scheduler if it is necessary.
        def run_validation(engine, validation_engine, valid_loader):
            validation_engine.run(valid_loader, max_epochs=1)

            if engine.lr_scheduler is not None:
                engine.lr_scheduler.step()

        # Attach above call-back function.
        train_engine.add_event_handler(
            Events.EPOCH_COMPLETED,
            run_validation,
            validation_engine,
            valid_loader
        )
        # Attach other call-back function for initiation of the training.
        train_engine.add_event_handler(
            Events.STARTED,
            self.target_engine_class.resume_training,
            self.config.init_epoch,
        )

        # Attach validation loss check procedure for every end of validation epoch.
        validation_engine.add_event_handler(
            Events.EPOCH_COMPLETED, self.target_engine_class.check_best
        )
        # Attach model save procedure for every end of validation epoch.
        validation_engine.add_event_handler(
            Events.EPOCH_COMPLETED,
            self.target_engine_class.save_model,
            train_engine,
            self.config,
            src_vocab,
            tgt_vocab,
        )

        # Start training.
        train_engine.run(train_loader, max_epochs=n_epochs)

        return model
