# SpiRob MuJoCo バックエンド

Web シミュレータ（`src/physics.ts`）の離散 Cosserat ロッドモデルを MuJoCo 3 に移植した物理バックエンドです。TS 版がスクリプト制約で作っていた巻き付き・接触を、腱（tendon）と接触力学から創発させることを目的としています。弾性物体（MuJoCo flex による四面体ソフトボディ）の把持シミュレーションに対応します。

## セットアップと実行

```bash
cd backend
uv sync

# テスト
uv run pytest

# 組み込みビューア（モデル検証用）
uv run spirob-viewer                       # 無張力でドラッグ観察
uv run spirob-viewer --forces 1.8 0 0      # 定張力
uv run spirob-viewer --grasp               # 自動把持シーケンス
uv run spirob-viewer --object soft_sphere --grasp   # 弾性球の把持

# FastAPI サーバ（WebSocket 状態配信）
uv run spirob-server --port 8000
```

## モデル化

- `unit_data.py` — TS 版と同一の 20 ユニット STL 寸法テーブル
- `model.py` — MJCF 生成。関節は Y/Z ヒンジ対（±28.8°、剛性は `EI/ℓ` の r³ 相似則）、3 本のケーブルは断面 120° 配置のサイトを経由する spatial tendon + 力制御モータ
- 弾性物体は `flexcomp`（四面体格子、Young 率 / Poisson 比を設定可）。flex は陽的積分のため timestep を 0.5 ms に自動短縮（減衰パラメータは MuJoCo 3.11 では発散するため 0 固定）
- 座標系は MuJoCo の Z-up。フロントエンド（Three.js, Y-up）へは `(x, y, z)_three = (x, z, -y)_mujoco` で変換する

## 自動把持：論文 Fig. 3A の忠実な再現（結果は物理に委ねる）

`grasp.py` は論文 Fig. 3A の 5 段階シーケンスを、論文の 3-cable デモ（「2D 把持と同様の戦略」）どおりに「ケーブル0 = パック側 F1、ケーブル1+2 の合力 = 反対側 F2」へ写像して実装しています：PACK（F1 を 6 N へ、螺旋パック）→ REACH（F2 を 3 N へ、基部からほどいて接近）→ WRAP（F1 を 6→2 N、先端螺旋を表面へ展開）→ GRASP（F2 を 6 N へ）→ HOLD（t=12 s で提示リグが手を放す）。張力の数値は論文に記載がない（3-cable 機は手動操作）ため本モデルの校正値です。これを実物理で成立させる構成：

1. **平面マウント（planar）＝ Fig. 3A の実配置** — 論文 Fig. 3A の写真は**真上から撮ったトップビュー**です：アームは滑らかなテーブル上で**水平面内**にカールし（重力は曲げ平面に垂直で、螺旋を潰さない）、木の丸棒は手で垂直に差し出されます。planar マウントはこれを再現します：基部クォータニオンがケーブル0 を世界 +Y に向け、螺旋は水平面内で +Y 側へ巻きます。テーブル（床）は priority=1 の低摩擦 geom（μ=0.25）で、アームは論文どおり面上を滑りながらカールします。既定 `mount='planar'`（`standing` / `hanging` / `horizontal` は探索用に残置）
2. **手（mocap + weld）による提示** — 丸棒は mocap ボディ「hand」に weld された剛体で、PACK 中はアーム前方 +X に退避（パッキングの掃引域に干渉しない）、REACH 中に把持位置 (0.15, 0.02) へ運ばれます — 論文で実験者の手が棒をアームへ寄せる操作そのものです。t=12 s に weld を解いて手を放し、以後は巻き付きだけが棒を支えます
3. **キャプスタン減衰と逐次変形** — 各ケーブルを関節ごとの 19 セグメント腱に分割し、毎サブステップ `T_i = T₀·exp(-μΣ|Δθ|)` で張力を配分（`simulation.py`）。REACH で新たに張る F2 がパック済み先端へ届かず基部から順にほどける「逐次変形」が論文の核心機構で、これには μ≥0.25 程度が必要（既定 μ=0.4。web 版の 0.08 では 2π 曲げでも先端に 6 割届いてしまい、REACH 中に先端螺旋が消失する）

**基部剛性 0.7 Nm/rad はこのバックエンドで再校正した値**です。TS 版の 0.09 では論文の 6 N パックで腕全体が基部まで丸まり、Fig. 3A の「先端だけ螺旋・基部直立」と矛盾します（TS 版はスクリプト拘束が形を作っていたため露見しなかった）。

### 現状の結果（正直な報告）

既定の planar シナリオ（30 mm 丸棒）は論文 Fig. 3A の各フレームを再現します：PACK で先端螺旋が形成され、REACH でアームが基部からほどけて伸び、手が運び込んだ棒がアーム側面に**接触**（Contact）、WRAP で先端螺旋が棒を乗り越えて転がり（Climbing）、GRASP で先端が棒に巻き付いて 3〜4 ユニットが同時接触します（Wrapping & grasping）。ただし **t=12 s に手を放すと棒は軸方向に滑り落ちます**：20 ユニットのアームでは 30 mm 棒の周囲に約 3/4 巻しか確保できず（論文の 2D 機は遠位がはるかに細かく多巻できる）、この接触数では 70 g の棒の軸方向保持に足りません。なお論文の Fig. 3A も手を放す操作は行っていません（最終フレームでも手が棒を保持）。現キャリブレーションに対する物理の答えは「この巻数ではこの棒は保持できない」であり、それを観測可能にするのがこのバックエンドの存在意義です。感度を見るための入口：

```bash
uv run spirob-viewer --grasp                     # 既定 30mm/70g 丸棒（planar、手が運び込む）
uv run spirob-viewer --grasp --size 22           # 細い棒（巻数が増え、接触 5 ユニット）
uv run spirob-viewer --grasp --mass 0.02         # 軽くしてみる
uv run spirob-viewer --grasp --stiffness 0.4     # 剛性校正の影響
uv run spirob-viewer --grasp --object soft_sphere --young 8e3   # 柔らかい対象（床置き）
uv run spirob-viewer --grasp --mount standing    # 旧・鉛直面シナリオ（吊り紐提示）
```

テスト（`test_paper_sequence_reaches_object`）は「安定に完走し、丸棒では 3 ユニット以上が接触する」ことまでを検証し、捕獲は要求していません。

## WebSocket プロトコル

`ws://host:port/ws` に接続すると `{"type": "meta", ...}`（ユニット寸法、flex 頂点数等）が届き、以後 60 Hz で `{"type": "state", ...}`（各ユニットの pos/quat、対象の pos/quat または flex 頂点座標、ケーブル張力、接触ユニット、把持フェーズ）が流れます。クライアントからは `cableForces` / `autoGrasp` / `reset` / `object` コマンドを JSON で送信します。詳細は `server.py` の docstring を参照。
