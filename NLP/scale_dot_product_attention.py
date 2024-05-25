"""
inplementation of scaled dot-product attention from https://github.com/Kyubyong/transformer
"""
import tensorflow as tf

def scaled_dot_product_attention(Q, K, V, key_masks,
                                 causality=False, dropout_rate=0.,
                                 training=True,
                                 scope="scaled_dot_product_attention"):
    '''
    Q: クエリをパックしたもの。3次元テンソル。[N, T_q, d_k]。
    K: キーをパックしたもの。3次元テンソル。[N, T_k, d_k]。
    V: 値をパックしたもの。3次元テンソル。[N, T_k, d_v]。
    key_masks: [N, key_seqlen]の形状を持つ2次元テンソル。
    causality: Trueの場合、将来の情報を隠すためのマスキングを適用。
    dropout_rate: [0, 1]の浮動小数点数。
    training: ドロップアウトを制御するためのブール値。
    scope: オプションの`variable_scope`。
    '''
    with tf.variable_scope(scope, reuse=tf.AUTO_REUSE):
        d_k = Q.get_shape().as_list()[-1] # クエリの最終次元のサイズを取得

        # 内積の計算
        outputs = tf.matmul(Q, tf.transpose(K, [0, 2, 1]))  # (N, T_q, T_k) クエリとキーの転置行列との行列積

        # スケーリング
        outputs /= d_k ** 0.5

        # キーマスキング
        outputs = mask(outputs, key_masks=key_masks, type="key") # キーマスクを適用

        # 因果性マスキングまたは未来のブラインディング
        if causality:
            outputs = mask(outputs, type="future") # 因果性マスクを適用

        # ソフトマックス
        outputs = tf.nn.softmax(outputs) # ソフトマックス関数を適用して正規化
        attention = tf.transpose(outputs, [0, 2, 1]) # attention行列を転置
        tf.summary.image("attention", tf.expand_dims(attention[:1], -1)) # TensorBoard用に注意行列の画像を出力

        # # クエリマスキング
        # outputs = mask(outputs, Q, K, type="query")

        # ドロップアウト
        outputs = tf.layers.dropout(outputs, rate=dropout_rate, training=training)

        # 重み付け和 (コンテキストベクトル)
        outputs = tf.matmul(outputs, V)  # (N, T_q, d_v) 出力と値の行列積を計算

    return outputs