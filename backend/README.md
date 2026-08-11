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

## 自動把持（現状の到達点と既知の課題）

`grasp.py` のスケジュールは TS 版の移植ではありません。実物理では張力レンジが全く異なるため、掃引実験で得た次の知見に基づきます：

- パック張力 ~2 N 超でコイル全体が基部側へ退縮し、対象から剥がれる（巻き付け形成は低張力で行う）
- 剛体対象を提示治具（weld）に固定したまま解放すると、蓄積接触力で対象が射出される → 解放時はフェードアウトする粘性ダンパで吸収
- 弾性球は重心のみバネ保持（変形自由）で提示すると、コイルが表面に食い込み最大 8 ユニット・40 点超の面接触で包み込める（剛体球は点接触 ~5 ユニット）

**既知の課題**: 解放後に対象の全重量を巻き付けだけで保持し続ける段階は未達成です。r³ 則に従う遠位関節は 45 g の荷重に対して柔らかすぎ、コイルの「口」が開いて対象が抜け落ちます（剛体球はコイル軸方向へ転がり抜け、円柱は軸方向に滑る）。保持まで到達するには、巻き数を増やせる小径対象、キャプスタン減衰を再現する分割腱、吊り下げマウントなどが候補です。`spirob-viewer --grasp` で挙動を確認しながら調整してください。

## WebSocket プロトコル

`ws://host:port/ws` に接続すると `{"type": "meta", ...}`（ユニット寸法、flex 頂点数等）が届き、以後 60 Hz で `{"type": "state", ...}`（各ユニットの pos/quat、対象の pos/quat または flex 頂点座標、ケーブル張力、接触ユニット、把持フェーズ）が流れます。クライアントからは `cableForces` / `autoGrasp` / `reset` / `object` コマンドを JSON で送信します。詳細は `server.py` の docstring を参照。
