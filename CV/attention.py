import torch
from torch import nn

class Attention(nn.Module):
    def __init__(self, dim, heads = 8, dim_head = 64, dropout = 0.):
        super().__init__()
        inner_dim = dim_head *  heads  # 内部次元を定義（ヘッド数 × 各ヘッドの次元）
        project_out = not (heads == 1 and dim_head == dim)  # 出力プロジェクションが必要かを判定

        self.heads = heads  # ヘッド数を保存
        self.scale = dim_head ** -0.5  # スケールファクタを定義（通常は次元の平方根の逆数）

        self.norm = nn.LayerNorm(dim)  # 入力を正規化するレイヤー

        self.attend = nn.Softmax(dim = -1)  # ソフトマックス関数（最後の次元で正規化）
        self.dropout = nn.Dropout(dropout)  # ドロップアウトを定義

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias = False)  # クエリ、キー、バリューを一度に計算するための線形変換

        self.to_out = nn.Sequential(  # 出力プロジェクションを定義（必要な場合のみ）
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()  # プロジェクションが不要な場合は恒等関数

    def forward(self, x):
        x = self.norm(x)  # 入力を正規化

        qkv = self.to_qkv(x).chunk(3, dim = -1)  # クエリ、キー、バリューを分割
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = self.heads), qkv)  # 各ヘッドに分割し並び替え

        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale  # クエリとキーの内積を計算し、スケーリング

        attn = self.attend(dots)  # ソフトマックスを適用し、注意重みを計算
        attn = self.dropout(attn)  # ドロップアウトを適用

        out = torch.matmul(attn, v)  # 注意重みとバリューの内積を計算
        out = rearrange(out, 'b h n d -> b n (h d)')  # 元の形に並び替え
        return self.to_out(out)  # 出力プロジェクションを適用し、出力を返す