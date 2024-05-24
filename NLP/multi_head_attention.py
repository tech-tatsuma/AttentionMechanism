from torch import nn

from NLP.scale_dot_product_attention import ScaleDotProductAttention


class MultiHeadAttention(nn.Module):

    def __init__(self, d_model, n_head):
        super(MultiHeadAttention, self).__init__()
        self.n_head = n_head # ヘッド数を保存
        self.attention = ScaleDotProductAttention() # スケールドットプロダクトアテンションのインスタンスを生成
        self.w_q = nn.Linear(d_model, d_model) # クエリの線形変換用の重み
        self.w_k = nn.Linear(d_model, d_model) # キーの線形変換用の重み
        self.w_v = nn.Linear(d_model, d_model) # バリューの線形変換用の重み
        self.w_concat = nn.Linear(d_model, d_model) # 最終的な出力の線形変換用の重み

    def forward(self, q, k, v, mask=None):
        # 1. 重み行列を使用してドットプロダクトを計算
        q, k, v = self.w_q(q), self.w_k(k), self.w_v(v)

        # 2. テンソルをヘッド数で分割
        q, k, v = self.split(q), self.split(k), self.split(v)

        # 3. スケールドットプロダクトアテンションを計算して類似性を求める
        out, attention = self.attention(q, k, v, mask=mask)

        # 4. 結果を結合して線形層に通す
        out = self.concat(out)
        out = self.w_concat(out)

        return out

    def split(self, tensor):
        """
        テンソルをヘッドの数で分割する

        :param tensor: [batch_size, length, d_model]
        :return: [batch_size, head, length, d_tensor]
        """
        batch_size, length, d_model = tensor.size() # テンソルのサイズを取得

        d_tensor = d_model // self.n_head # 各ヘッドの次元数を計算
        tensor = tensor.view(batch_size, length, self.n_head, d_tensor).transpose(1, 2) # テンソルをヘッドの数に基づいて分割し，次元を入れ替える

        return tensor

    def concat(self, tensor):
        """
        self.split(tensor : torch.Tensor)の逆関数

        :param tensor: [batch_size, head, length, d_tensor]
        :return: [batch_size, length, d_model]
        """
        batch_size, head, length, d_tensor = tensor.size() # テンソルのサイズを取得
        d_model = head * d_tensor # 元の次元数を計算

        tensor = tensor.transpose(1, 2).contiguous().view(batch_size, length, d_model) #次元を元に戻してテンソルを連結
        return tensor