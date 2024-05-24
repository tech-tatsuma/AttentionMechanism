import math

from torch import nn


class ScaleDotProductAttention(nn.Module):
    """
    Query : 注目している文(Decoder)
    Key : Queryとの関係を調べる全ての文(Encoder)
    Value : Keyと同じ全ての文(Encoder)
    """

    def __init__(self):
        super(ScaleDotProductAttention, self).__init__()
        self.softmax = nn.Softmax(dim=-1) # USER:関数の定義

    def forward(self, q, k, v, mask=None, e=1e-12):
        # 入力は4次元テンソルを想定
        # [バッチサイズ, ヘッド数, 長さ, テンソルの次元数]
        batch_size, head, length, d_tensor = k.size()

        # 1. QueryとKey^Tのドット積（内積）を計算して類似度を求める
        k_t = k.transpose(2, 3)  # 転置処理
        score = (q @ k_t) / math.sqrt(d_tensor)  # scaled dot product

        # 2. マスクの適用(任意)
        if mask is not None:
            score = score.masked_fill(mask == 0, -10000)

        # 3. USER:を通して[0,1]の範囲にする
        score = self.softmax(score)

        # 4. Valueと掛け算をする
        v = score @ v

        return v, score # アテンションの結果とスコアを返す