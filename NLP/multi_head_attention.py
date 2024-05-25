import tensorflow as tf

def multihead_attention(queries, keys, values, key_masks,
                        num_heads=8, 
                        dropout_rate=0,
                        training=True,
                        causality=False,
                        scope="multihead_attention"):
    '''
    マルチヘッドアテンションを適用します。3.2.2を参照。
        queries: 形状が[N, T_q, d_model]の3次元テンソル。
        keys: 形状が[N, T_k, d_model]の3次元テンソル。
        values: 形状が[N, T_k, d_model]の3次元テンソル。
        key_masks: 形状が[N, key_seqlen]の2次元テンソル。
        num_heads: 整数。ヘッドの数。
        dropout_rate: 浮動小数点数。ドロップアウト率。
        training: ブール値。ドロップアウトの制御用。
        causality: ブール値。Trueの場合、未来の情報を参照するユニットがマスクされます。
        scope: オプションの`variable_scope`。

        戻り値
        形状が(N, T_q, C)の3次元テンソル
    '''
    d_model = queries.get_shape().as_list()[-1] # クエリの最終次元のサイズを取得
    with tf.variable_scope(scope, reuse=tf.AUTO_REUSE):
        # 線形変換
        Q = tf.layers.dense(queries, d_model, use_bias=True) # (N, T_q, d_model) クエリを線形変換
        K = tf.layers.dense(keys, d_model, use_bias=True) # (N, T_k, d_model) キーを線形変換
        V = tf.layers.dense(values, d_model, use_bias=True) # (N, T_k, d_model) 値を線形変換
        
        # 分割して結合
        Q_ = tf.concat(tf.split(Q, num_heads, axis=2), axis=0) # (h*N, T_q, d_model/h) Qをヘッド数に分割し結合
        K_ = tf.concat(tf.split(K, num_heads, axis=2), axis=0) # (h*N, T_k, d_model/h) Kをヘッド数に分割し結合
        V_ = tf.concat(tf.split(V, num_heads, axis=2), axis=0) # (h*N, T_k, d_model/h) Vをヘッド数に分割し結合

        # Attention
        outputs = scaled_dot_product_attention(Q_, K_, V_, key_masks, causality, dropout_rate, training) # スケール付きドットプロダクトアテンションを適用

        # 形状を元に戻す
        outputs = tf.concat(tf.split(outputs, num_heads, axis=0), axis=2 ) # (N, T_q, d_model) ヘッド数に分割された出力を結合し元の形状に戻す
              
        # 残差接続
        outputs += queries # 残差接続を適用
              
        # 正規化
        outputs = ln(outputs) # レイヤー正規化を適用
 
    return outputs