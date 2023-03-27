import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence as pack
from torch.nn.utils.rnn import pad_packed_sequence as unpack

import sys
sys.path.append('/content/drive/MyDrive/인공지능/기계번역/simple-nmt')
import simple_nmt.data_loader as data_loader
from simple_nmt.search import SingleBeamSearchBoard


class Attention(nn.Module): # 쿼리를 변환하는 방법(linear transformation)

    def __init__(self, hidden_size):
        super(Attention, self).__init__()

        self.linear = nn.Linear(hidden_size, hidden_size, bias=False) # linear transformation하기 위해 선언 
        self.softmax = nn.Softmax(dim=-1) # q X k에대가 softmax를 씌워준다./그리고나면 attention weight가 나온다./그래서 미리 선언

        # W = Softmax(Q*W*Kt)->q,w,kt사이에 마스크를 씌워줘서 pad를 -inf로 보내주는 작업이 경우에 다라 필요하다.
        # c = W*V

    def forward(self, h_src, h_t_tgt, mask=None):
        # |h_src| = (batch_size, length, hidden_size)
        # |h_t_tgt| = (batch_size, 1, hidden_size)
        # |mask| = (batch_size, length)

        query = self.linear(h_t_tgt) # Q*W
        # |query| = (batch_size, 1, hidden_size)

        # length, hidden_size를 transpose 즉 Kt(bs,hs,n) 이거를 query와 곱한다.
        # q*kt = (bs,1,hs) * (bs,hs,n) = (bs,1,n) = (미니배치의 각 샘플별, 디코더의 현재 타임스텝에 대해서, 인코더의 전체 타임스텝에 대해서) weight값이 들어있는 텐서이다.
        weight = torch.bmm(query, h_src.transpose(1, 2)) 
        # |weight| = (batch_size, 1, length)
        if mask is not None:
            # Set each weight as -inf, if the mask value equals to 1.
            # Since the softmax operation makes -inf to 0,
            # masked weights would be set to 0 after softmax operation.
            # Thus, if the sample is shorter than other samples in mini-batch,
            # the weight for empty time-step would be set to 0.
            weight.masked_fill_(mask.unsqueeze(1), -float('inf')) # length mask위치에 true가 있는 위치에 -무한대를 줘라
        weight = self.softmax(weight) # softmax를 취해준다.

        context_vector = torch.bmm(weight, h_src) # 행렬의 배치 행렬-행렬 곱을 수행
        # |context_vector| = (batch_size, 1, hidden_size)

        return context_vector


class Encoder(nn.Module):

    def __init__(self, word_vec_size, hidden_size, n_layers=4, dropout_p=.2):
        super(Encoder, self).__init__()

        # Be aware of value of 'batch_first' parameter.
        # Also, its hidden_size is half of original hidden_size,
        # because it is bidirectional.
        self.rnn = nn.LSTM(
            word_vec_size, ## input_size
            int(hidden_size / 2), # hidden_size : bidirectionally 정방향 역방향 각각 줄것이기 때문에 2로 나눈다.
            num_layers=n_layers, 
            dropout=dropout_p,# LSTM사이층에 들어갈 dropout
            bidirectional=True, 
            batch_first=True, # batch를 shape의 첫번째로?
        )

    def forward(self, emb): # 임베딩된 텐서를 받아서 
        # |emb| = (batch_size, length, word_vec_size)

        if isinstance(emb, tuple): # 임베딩 텐서가 튜플인지? / padded squence의 타입의 객체로 변환 / 그것을 맵핑하는 작업
            x, lengths = emb
            x = pack(x, lengths.tolist(), batch_first=True) 
            # 튜플이면 padded squence로 갈것이고
            # Below is how pack_padded_sequence works.
            # As you can see,
            # PackedSequence object has information about mini-batch-wise information,
            # not time-step-wise information.
            # 
            # a = [torch.tensor([1,2,3]), torch.tensor([3,4])]
            # b = torch.nn.utils.rnn.pad_sequence(a, batch_first=True)
            # >>>>
            # tensor([[ 1,  2,  3],
            #     [ 3,  4,  0]])
            # torch.nn.utils.rnn.pack_padded_sequence(b, batch_first=True, lengths=[3,2]
            # >>>>PackedSequence(data=tensor([ 1,  3,  2,  4,  3]), batch_sizes=tensor([ 2,  2,  1]))
        else: # 튜플이 아니면 그냥 그대로 
            x = emb

        y, h = self.rnn(x) # 출력과 hidden state(마지막 time step의 ) hidden과 sell의 튜플로 되어있다.
        # |y| = (batch_size, length, hidden_size)
        # |h[0]| = (num_layers * 2, batch_size, hidden_size / 2) # sell는 h[1]

        if isinstance(emb, tuple):
            y, _ = unpack(y, batch_first=True) # pack를 해주었다면 unpack해준다.

        return y, h


class Decoder(nn.Module):

    def __init__(self, word_vec_size, hidden_size, n_layers=4, dropout_p=.2):
        super(Decoder, self).__init__()

        # Be aware of value of 'batch_first' parameter and 'bidirectional' parameter.
        self.rnn = nn.LSTM( # 유니디렉션LSTM을 만들어야 함
            word_vec_size + hidden_size,  
            hidden_size,
            num_layers=n_layers,
            dropout=dropout_p,
            bidirectional=False,
            batch_first=True,
        )

    # emb_t와 h_t-1_tilde, h_t-1(이전 타임스텝의 ht-1과 ct-1이 같이 들어있음 )
    def forward(self, emb_t, h_t_1_tilde, h_t_1): # 인코더와 달리 한 타임스텝씩 들어올 것이다./input peding때문에, 추론할때도 당연히
        # |emb_t| = (batch_size, 1, word_vec_size) -> 현재 타임스텝이라서 1, 그리고 배치사이즈와 워드임베딩
         # |h_t_1_tilde| = (batch_size, 1, hidden_size) -> 
        # |h_t_1[0]| = (n_layers, batch_size, hidden_size) -> 이전 타임스텝의 LSTM의 hidden_state와 cell_state의 튜플
        batch_size = emb_t.size(0) # batch-size
        hidden_size = h_t_1[0].size(-1) # hidden_state의 hidden_size

        if h_t_1_tilde is None: # None이면 첫번째 타임스텝이다.
            # If this is the first time-step,
            h_t_1_tilde = emb_t.new(batch_size, 1, hidden_size).zero_() # new를 써서emb_t와 같은 디바이스와 타입 사이즈로 만들어라 그리고 0으로 채워라/즉 초기화

        # Input feeding trick
        x = torch.cat([emb_t, h_t_1_tilde], dim=-1) # rnn에 넣기전에 마지막 차원에 대해서 붙인다
        # (bs, 1, ws+hs)가 된다.

        # Unlike encoder, decoder must take an input for sequentially.
        y, h = self.rnn(x, h_t_1)
        # y = (bs, 1, hs)
        # h[0] = h[1] = (n_layers, bs, hs)

        return y, h
        # 이게 나오면 여기에다가 attention을준다. attention을 하고  나면 cv가 나온다.
        # cv와 현재 타임스텝의 y를 cat해서 h_tilde를 구한다. 
        # 다음 타임 스텝에 h_tilde와 그것을 generator에 가서 값이 나올것인데
        # embed_layer에 거쳐서 word_emb_vec와 입력으로 들어옴


class Generator(nn.Module): # 학습할때는 모든 time_step를 한번에 통과 추론할때는 하나씩 통과

    def __init__(self, hidden_size, output_size):
        super(Generator, self).__init__()

        self.output = nn.Linear(hidden_size, output_size) # linear로 softmax를 씌울 수 있도록 변환
        self.softmax = nn.LogSoftmax(dim=-1) # log 확률을 뱉어내도록 한다. output 사이즈 dm에 대해 동작하도록 선언함함

    def forward(self, x):
        # |x| = (batch_size, length, hidden_size)

        y = self.softmax(self.output(x)) 
        # |y| = (batch_size, length, output_size)

        # Return log-probability instead of just probability.
        return y


class Seq2Seq(nn.Module):

    def __init__(
        self,
        input_size,
        word_vec_size,
        hidden_size,
        output_size,
        n_layers=4,
        dropout_p=.2
    ):
        self.input_size = input_size
        self.word_vec_size = word_vec_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.n_layers = n_layers
        self.dropout_p = dropout_p

        super(Seq2Seq, self).__init__()

        self.emb_src = nn.Embedding(input_size, word_vec_size)
        self.emb_dec = nn.Embedding(output_size, word_vec_size)

        self.encoder = Encoder(
            word_vec_size, hidden_size,
            n_layers=n_layers, dropout_p=dropout_p,
        )
        self.decoder = Decoder(
            word_vec_size, hidden_size,
            n_layers=n_layers, dropout_p=dropout_p,
        )
        self.attn = Attention(hidden_size)

        # attention cv와 decoder의 현재 타임스텝 output을 concat해서 h_tilde를 만들어야 된다.
        # concat하는 입력을 받아서 hidden_size로 바꿔주는 레이어를 만들고
        # 거기에 씌워줄 tanh를 선언
        # 그리고 generator h_dilte를 받을거니 hidden_size를 받아서 taget의 단어장 사이즈로 바꾸준다.
        self.concat = nn.Linear(hidden_size * 2, hidden_size) 
        self.tanh = nn.Tanh()
        self.generator = Generator(hidden_size, output_size)

    # |length| = (bs,)
    def generate_mask(self, x, length):
        mask = []

        max_length = max(length)
        for l in length:
            if max_length - l > 0: # 가장 긴 길이가 아닐경우우
                # If the length is shorter than maximum length among samples, 
                # set last few values to be 1s to remove attention weight.
                mask += [torch.cat([x.new_ones(1, l).zero_(), # 필요한만큼 0으로
                                    x.new_ones(1, (max_length - l)) # 남은 것을 1로 채워준다.
                                    ], dim=-1)] # 이것을 concat한다. -1 차원에서서
            else:
                # If the length of the sample equals to maximum length among samples, 
                # set every value in mask to be 0.
                mask += [x.new_ones(1, l).zero_()] # 0으로 꽉채움움

        mask = torch.cat(mask, dim=0).bool() # 각각의 것을 만들고 나면 리스트를 0차원에 대해서 cat한다. 그리고 bool으로 선언언

        return mask # x와 tensor와 lenth가 주어졌을때 pad의 모양으로 mask를 만들 수

    # encoder hiddens를 받아서 decoder hiddens로 치환하는 것이다.
    def merge_encoder_hiddens(self, encoder_hiddens):
        new_hiddens = []
        new_cells = []

        # |hiddens| = (Nlayers * 2, bs, hs/2)
        hiddens, cells = encoder_hiddens

        # i-th and (i+1)-th layer is opposite direction.
        # Also, each direction of layer is half hidden size.
        # Therefore, we concatenate both directions to 1 hidden size layer.
        for i in range(0, hiddens.size(0), 2): # Nlayer * 2를 두칸씩 띄워서서
            # 0,1 concat 2,3concat 4,5concat (hs에 대해서)
            # (bs, hs/2)인 상태에 X 2로 되어있을 것이다. 즉 (bs,hs)가 된다.
            # X 2인 이유는 2칸씩 띄워서일까?
            new_hiddens += [torch.cat([hiddens[i], hiddens[i + 1]], dim=-1)] 
            new_cells += [torch.cat([cells[i], cells[i + 1]], dim=-1)]

        # stack을 하게 되면 Nlayers, bs, hs로 나오게 된다.
        new_hiddens, new_cells = torch.stack(new_hiddens), torch.stack(new_cells)

        return (new_hiddens, new_cells)

    def fast_merge_encoder_hiddens(self, encoder_hiddens):
        # Merge bidirectional to uni-directional
        # We need to convert size from (n_layers * 2, batch_size, hidden_size / 2) 이거를 바로 아래모양처럼 만들어야 한다.
        # to (n_layers, batch_size, hidden_size).
        # Thus, the converting operation will not working with just 'view' method.
        h_0_tgt, c_0_tgt = encoder_hiddens
        batch_size = h_0_tgt.size(1)
        # |h0 tgt| = (Nlayer*2, bs , hs/2) -> transpose후 (bs, Nlayers*2, hs/2) -> view를 해주면 (bs, Nlayer, hs) -> transpose후 (Nlayers, bs, hs)
        # 왜 바로 view하지 않고 transpose를 두번했을까? view로 바로 못바꿀 것이다. shape는 나올 것이지만 안에 구성이 이상하게 바꾸니다.
        h_0_tgt = h_0_tgt.transpose(0, 1).contiguous().view(batch_size,
                                                            -1,
                                                            self.hidden_size
                                                            ).transpose(0, 1).contiguous()
        c_0_tgt = c_0_tgt.transpose(0, 1).contiguous().view(batch_size,
                                                            -1,
                                                            self.hidden_size
                                                            ).transpose(0, 1).contiguous()
        # You can use 'merge_encoder_hiddens' method, instead of using above 3 lines.
        # 'merge_encoder_hiddens' method works with non-parallel way.
        # h_0_tgt = self.merge_encoder_hiddens(h_0_tgt)

        # |h_src| = (batch_size, length, hidden_size)
        # |h_0_tgt| = (n_layers, batch_size, hidden_size)
        return h_0_tgt, c_0_tgt

    # |src| = (bs, n) ≈ (bs, n, |Vsrc|) ≈는 근사함
    # |tgt| = (bs, m) ≈ (bs, m, |Vtgt|)
    # |output| = (bs, m, |Vtgt|)
    
    def forward(self, src, tgt):
        batch_size = tgt.size(0)

        mask = None
        x_length = None
        if isinstance(src, tuple):
            x, x_length = src
            # Based on the length information, gererate mask to prevent that
            # shorter sample has wasted attention.
            mask = self.generate_mask(x, x_length)
            # |mask| = (batch_size, length)
        else:
            x = src

        if isinstance(tgt, tuple):
            tgt = tgt[0]

        # Get word embedding vectors for every time-step of input sentence.
        emb_src = self.emb_src(x)
        # |emb_src| = (batch_size, length, word_vec_size)

        # The last hidden state of the encoder would be a initial hidden state of decoder.
        h_src, h_0_tgt = self.encoder((emb_src, x_length))
        # |h_src| = (batch_size, length, hidden_size) --> 인코더의 전체 타임스텝 마지막레이어의 hs
        # |h_0_tgt| = (n_layers * 2, batch_size, hidden_size / 2) --> 인코더의 마지막 타임스텝의 전체 레이어의 hs

        h_0_tgt = self.fast_merge_encoder_hiddens(h_0_tgt)
        emb_tgt = self.emb_dec(tgt) # 정답을 한꺼번에 emb_vecter로 만들어줌
        # |emb_tgt| = (batch_size, length, word_vec_size)
        h_tilde = []

        h_t_tilde = None # 첫번째 time_step이므로 이전 tiem_step은 None
        decoder_hidden = h_0_tgt # 이전의 decoder의 hs는 encoder의 마지막 hs로 만들었다. 
        # Run decoder until the end of the time-step.
        for t in range(tgt.size(1)): # target의 길이만큼 for문을 돈다.
            # Teacher Forcing: take each input from training set,
            # not from the last time-step's output.
            # Because of Teacher Forcing,
            # training procedure and inference procedure becomes different.
            # Of course, because of sequential running in decoder,
            # this causes severe bottle-neck.
            # bs와 wvs다 가져오고 length는 t번째 가져옴
            # unsqueeze(1)을 해서 타임스텝을 만들어줬다.
            emb_t = emb_tgt[:, t, :].unsqueeze(1) 
            # |emb_t| = (batch_size, 1, word_vec_size)
            # |h_t_tilde| = (batch_size, 1, hidden_size) # 이전 타임스텝의 h_tilde

            # 현재 타임스텝의 출력, 현재 타임스텝의 hs, cs의 튜플
            decoder_output, decoder_hidden = self.decoder(emb_t, # 현재 타임스텝의 ev
                                                          h_t_tilde, # 지난 타임스텝의 h_tilde
                                                          decoder_hidden # 지난 타임스텝의 hs, cs의 튜플플
                                                          )
            # |decoder_output| = (batch_size, 1, hidden_size)
            # |decoder_hidden| = (n_layers, batch_size, hidden_size)

            context_vector = self.attn(h_src, decoder_output, mask) # 인코더의 전체 타임스텝의 output과 마스크를 넣는다. 
            # |context_vector| = (batch_size, 1, hidden_size)

            # concat layer에 넣기 전에 합친다.
            # [(bs, 1, hs), (bs, 1, hs)]의 -1 dim
            # (bs, 1 hs*2)
            # concat layer를 선언할때 hs*2를 입력으로 받도록했다.
            # 출력은 hs를 뱉는다.
            # tanh를 씌운다.
            h_t_tilde = self.tanh(self.concat(torch.cat([decoder_output, 
                                                         context_vector
                                                         ], dim=-1)))
            # |h_t_tilde| = (batch_size, 1, hidden_size)

            h_tilde += [h_t_tilde]

        h_tilde = torch.cat(h_tilde, dim=1) # for문돌아서 1차원 즉 1 -> (bs,m,hs)
        # |h_tilde| = (batch_size, length, hidden_size)

        y_hat = self.generator(h_tilde) # 한번에 통과
        # |y_hat| = (batch_size, length, output_size)

        return y_hat

    def search(self, src, is_greedy=True, max_length=255):
        if isinstance(src, tuple): # ped seq 다루는 부분
            x, x_length = src
            mask = self.generate_mask(x, x_length)
        else:
            x, x_length = src, None
            mask = None
        batch_size = x.size(0)

        # Same procedure as teacher forcing.
        # encoder는 양방향, 디코더는 한방향 그래서 shape가 안맞는 경우가 생간다.
        # 그래서 그것을 해결하기 위해서 merge encoder hiddens를 만들었다. shape 을 맞춰준다.
        emb_src = self.emb_src(x)
        h_src, h_0_tgt = self.encoder((emb_src, x_length))
        decoder_hidden = self.fast_merge_encoder_hiddens(h_0_tgt)

        # Fill a vector, which has 'batch_size' dimension, with BOS value.
        # 생성해야되니 BOS가 있어야함
        # x랑 같은 텐서 타입의 디바이스를 만드는데 x를 제로로 해주고, 
        # 여기에 bos를 더하면 bos 토큰이 되는 것이다.
        y = x.new(batch_size, 1).zero_() + data_loader.BOS

        is_decoding = x.new_ones(batch_size, 1).bool()
        h_t_tilde, y_hats, indice = None, [], []
        
        # Repeat a loop while sum of 'is_decoding' flag is bigger than 0,
        # or current time-step is smaller than maximum length.
        while is_decoding.sum() > 0 and len(indice) < max_length: # 하나라도 디코딩하면 계속돈다. 그리고 max_length보다 작은동안안
            # Unlike training procedure,
            # take the last time-step's output during the inference.
            emb_t = self.emb_dec(y)
            # |emb_t| = (batch_size, 1, word_vec_size)

            decoder_output, decoder_hidden = self.decoder(emb_t,
                                                          h_t_tilde,
                                                          decoder_hidden)
            context_vector = self.attn(h_src, decoder_output, mask)
            h_t_tilde = self.tanh(self.concat(torch.cat([decoder_output,
                                                         context_vector
                                                         ], dim=-1)))
            y_hat = self.generator(h_t_tilde)
            # |y_hat| = (batch_size, 1, output_size)
            y_hats += [y_hat]

            if is_greedy: # 그리디라면 가장 좋은 것을 뽑아라라
                y = y_hat.argmax(dim=-1)
                # |y| = (batch_size, 1)
            else: # 랜덤을 할것이면
                # Take a random sampling based on the multinoulli distribution.
                y = torch.multinomial(y_hat.exp().view(batch_size, -1), 1)
                # |y| = (batch_size, 1)

            # Put PAD if the sample is done.
            y = y.masked_fill_(~is_decoding, data_loader.PAD) # is_decoding가 false면 의미없는 작업이었고 pad다./이전타임스텝에 EOS가 나왔을경우
            # Update is_decoding if there is EOS token.
            is_decoding = is_decoding * torch.ne(y, data_loader.EOS) # 이번 타임스텝에 EOs가 나온경우/ 안나왓을경우 true 나오면 false 이것을 곱한다. and 연산자르 거는 것것
            # |is_decoding| = (batch_size, 1)
            indice += [y]

        y_hats = torch.cat(y_hats, dim=1)
        indice = torch.cat(indice, dim=1)
        # |y_hat| = (batch_size, length, output_size)
        # |indice| = (batch_size, length)

        return y_hats, indice

    #@profile
    def batch_beam_search(
        self,
        src,
        beam_size=5,
        max_length=255,
        n_best=1,
        length_penalty=.2
    ):
        mask, x_length = None, None

        if isinstance(src, tuple):
            x, x_length = src
            mask = self.generate_mask(x, x_length)
            # |mask| = (batch_size, length)
        else:
            x = src
        batch_size = x.size(0)

        emb_src = self.emb_src(x)
        h_src, h_0_tgt = self.encoder((emb_src, x_length))
        # |h_src| = (batch_size, length, hidden_size)
        h_0_tgt = self.fast_merge_encoder_hiddens(h_0_tgt)

        # initialize 'SingleBeamSearchBoard' as many as batch_size
        boards = [SingleBeamSearchBoard(
            h_src.device,
            {
                'hidden_state': {
                    'init_status': h_0_tgt[0][:, i, :].unsqueeze(1),
                    'batch_dim_index': 1,
                }, # |hidden_state| = (n_layers, batch_size, hidden_size)
                'cell_state': {
                    'init_status': h_0_tgt[1][:, i, :].unsqueeze(1),
                    'batch_dim_index': 1,
                }, # |cell_state| = (n_layers, batch_size, hidden_size)
                'h_t_1_tilde': {
                    'init_status': None,
                    'batch_dim_index': 0,
                }, # |h_t_1_tilde| = (batch_size, 1, hidden_size)
            },
            beam_size=beam_size,
            max_length=max_length,
        ) for i in range(batch_size)]
        is_done = [board.is_done() for board in boards]

        length = 0
        # Run loop while sum of 'is_done' is smaller than batch_size, 
        # or length is still smaller than max_length.
        while sum(is_done) < batch_size and length <= max_length:
            # current_batch_size = sum(is_done) * beam_size

            # Initialize fabricated variables.
            # As far as batch-beam-search is running, 
            # temporary batch-size for fabricated mini-batch is 
            # 'beam_size'-times bigger than original batch_size.
            fab_input, fab_hidden, fab_cell, fab_h_t_tilde = [], [], [], []
            fab_h_src, fab_mask = [], []
            
            # Build fabricated mini-batch in non-parallel way.
            # This may cause a bottle-neck.
            for i, board in enumerate(boards):
                # Batchify if the inference for the sample is still not finished.
                if board.is_done() == 0:
                    y_hat_i, prev_status = board.get_batch()
                    hidden_i    = prev_status['hidden_state']
                    cell_i      = prev_status['cell_state']
                    h_t_tilde_i = prev_status['h_t_1_tilde']

                    fab_input  += [y_hat_i]
                    fab_hidden += [hidden_i]
                    fab_cell   += [cell_i]
                    fab_h_src  += [h_src[i, :, :]] * beam_size
                    fab_mask   += [mask[i, :]] * beam_size
                    if h_t_tilde_i is not None:
                        fab_h_t_tilde += [h_t_tilde_i]
                    else:
                        fab_h_t_tilde = None

            # Now, concatenate list of tensors.
            fab_input  = torch.cat(fab_input,  dim=0)
            fab_hidden = torch.cat(fab_hidden, dim=1)
            fab_cell   = torch.cat(fab_cell,   dim=1)
            fab_h_src  = torch.stack(fab_h_src)
            fab_mask   = torch.stack(fab_mask)
            if fab_h_t_tilde is not None:
                fab_h_t_tilde = torch.cat(fab_h_t_tilde, dim=0)
            # |fab_input|     = (current_batch_size, 1)
            # |fab_hidden|    = (n_layers, current_batch_size, hidden_size)
            # |fab_cell|      = (n_layers, current_batch_size, hidden_size)
            # |fab_h_src|     = (current_batch_size, length, hidden_size)
            # |fab_mask|      = (current_batch_size, length)
            # |fab_h_t_tilde| = (current_batch_size, 1, hidden_size)

            emb_t = self.emb_dec(fab_input)
            # |emb_t| = (current_batch_size, 1, word_vec_size)

            fab_decoder_output, (fab_hidden, fab_cell) = self.decoder(emb_t,
                                                                      fab_h_t_tilde,
                                                                      (fab_hidden, fab_cell))
            # |fab_decoder_output| = (current_batch_size, 1, hidden_size)
            context_vector = self.attn(fab_h_src, fab_decoder_output, fab_mask)
            # |context_vector| = (current_batch_size, 1, hidden_size)
            fab_h_t_tilde = self.tanh(self.concat(torch.cat([fab_decoder_output,
                                                             context_vector
                                                             ], dim=-1)))
            # |fab_h_t_tilde| = (current_batch_size, 1, hidden_size)
            y_hat = self.generator(fab_h_t_tilde)
            # |y_hat| = (current_batch_size, 1, output_size)

            # separate the result for each sample.
            # fab_hidden[:, begin:end, :] = (n_layers, beam_size, hidden_size)
            # fab_cell[:, begin:end, :]   = (n_layers, beam_size, hidden_size)
            # fab_h_t_tilde[begin:end]    = (beam_size, 1, hidden_size)
            cnt = 0
            for board in boards:
                if board.is_done() == 0:
                    # Decide a range of each sample.
                    begin = cnt * beam_size
                    end = begin + beam_size

                    # pick k-best results for each sample.
                    board.collect_result(
                        y_hat[begin:end],
                        {
                            'hidden_state': fab_hidden[:, begin:end, :],
                            'cell_state'  : fab_cell[:, begin:end, :],
                            'h_t_1_tilde' : fab_h_t_tilde[begin:end],
                        },
                    )
                    cnt += 1

            is_done = [board.is_done() for board in boards]
            length += 1

        # pick n-best hypothesis.
        batch_sentences, batch_probs = [], []

        # Collect the results.
        for i, board in enumerate(boards):
            sentences, probs = board.get_n_best(n_best, length_penalty=length_penalty)

            batch_sentences += [sentences]
            batch_probs     += [probs]

        return batch_sentences, batch_probs