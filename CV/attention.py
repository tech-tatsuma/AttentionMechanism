from typing import List

import tensorflow as tf
from tensorflow.keras import Model, layers

class MultiHeadSelfAttention(layers.Layer):
    def __init__(
        self,
        hidden_dim: int,
        num_heads: int,
        drop_rate: float = 0.,
        **kwargs):
        super().__init__(**kwargs)

        self.hidden_dim = hidden_dim # 隠れ次元の設定
        self.num_heads = num_heads # ヘッド数の設定
        self.drop_rate = drop_rate # ドロップアウトの設定

    def build(self, input_shape: List[tf.TensorShape]) -> None: 
        # Denseレイヤーの定義
        self.d_query = layers.Dense(self.hidden_dim, name='MHA_query') # クエリ用のDenseレイヤー
        self.d_key = layers.Dense(self.hidden_dim, name='MHA_query') # キー用のDenseレイヤー
        self.d_value = layers.Dense(self.hidden_dim, name='MHA_query') # バリュー用のDenseレイヤー

        # 形状変換レイヤーの定義
        self.reshape = layers.Reshape(
                target_shape=(-1, self.num_heads, self.hidden_dim//self.num_heads)
                ) # マルチヘッド用に形状を変換
        # パーミュテーションレイヤーの定義
        self.perm = layers.Permute(dims = (2,1,3)) # テンソルの次元を入れ替える

        # softmaxレイヤーの定義
        self.softmax = layers.Activation(activation = 'softmax', name = 'qkt_softmax')

        # ドロップアウトレイヤーの定義
        self.dropout = layers.Dropout(rate = self.drop_rate) # ドロップアウト

        # 出力用の形状変換とDenseレイヤーの定義
        self.reshape_f = layers.Reshape(target_shape = (-1, self.hidden_dim)) # 元の形状に戻すためのReshape
        self.dense = layers.Dense(units = self.hidden_dim) # 出力用のDenseレイヤー

    def reshape_and_perm(self, inputs: tf.Tensor) -> tf.Tensor:
        x = self.reshape(inputs) # 形状変換
        return self.perm(x) #次元の入れ替え

    def call(self, inputs: List[tf.Tensor]) -> tf.Tensor:
        # Denseレイヤーを適用
        query = self.reshape_and_perm(self.d_query(inputs)) # クエリの変換
        key = self.reshape_and_perm(self.d_key(inputs)) # キーの変換
        value = self.reshape_and_perm(self.d_value(inputs)) # バリューの変換

        # qkt = QK^T
        qkt = tf.matmul(a = query, b = key, transpose_b = True) # クエリとキーの行列積

        # qkt = softmax(qkt/sqrt(dim))
        d_k = tf.cast(self.hidden_dim, dtype=qkt.dtype) # 隠れ次元をテンソルの型にキャスト
        qkt = self.softmax(qkt / tf.sqrt(d_k)) # スケーリングしてソフトマックスを適用

        # ドロップアウトの適用
        qkt = self.dropout(qkt) # ドロップアウト

        # qktV
        attn = qkt @ value # スコアとバリューの行列積(アテンションスコアの計算)

        # [Batch, num_heads, patch^2+1, dim // num_heads] -> [Batch, patch^2+1, dim]
        attn = self.perm(attn) # 次元の入れ替え
        attn = self.reshape_f(attn) # 元の形状に戻す

        # 最終的なDenseレイヤーの適用（マルチヘッドアテンションを統合）
        out = self.dense(attn) # Denseレイヤーを適用して出力

        return out