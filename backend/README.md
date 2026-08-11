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

`grasp.py` は論文（と TS 版 `autoGraspCommand`）の 5 段階シーケンスをそのまま実装しています：PACK（パック側 6 N、先端から螺旋パック）→ REACH（反対側 2 本を 5.82 N へ、基部からほどいて接近）→ WRAP（パック側 6→5.2 N、表面登り）→ GRASP（反対側 9 N で摩擦閉包）→ HOLD。スケジュールは脚色していません。これを実物理で成立させるための構成が3つ：

1. **キャプスタン減衰** — 各ケーブルを関節ごとの 19 セグメント腱に分割し、毎サブステップ `T_i = T₀·exp(-μΣ|Δθ|)` で張力を配分（`simulation.py`）。連続腱では張力が経路全体で均一になり、パック／アンワインドの非対称性が原理的に消えるため
2. **吊り下げマウント** — Fig. 3A は垂下姿勢の動作。既定 `mount='hanging'`（`--mount horizontal` で水平化）
3. **物理的な提示ステージ** — 対象はレール上のサーボ台座（slide joint + position servo）に載り、REACH 中に横からスライドインし HOLD 中に退出する。摩擦だけで運ばれ、見えない力は一切ない（剛体球が台から転げ落ちないよう condim=6 + 転がり摩擦を設定）

**基部剛性 0.7 Nm/rad はこのバックエンドで再校正した値**です。TS 版の 0.09 では論文の 6 N パックで腕全体が基部まで丸まり、Fig. 3A の「先端だけ螺旋・基部直立」と矛盾します（TS 版はスクリプト拘束が形を作っていたため露見しなかった）。

### 現状の結果（正直な報告）

52 mm・45 g の既定条件では、螺旋降下と表面接触（WRAP 中に1〜2ユニット）までは論文どおり進みますが、**把持閉包には至らず対象は捕獲されません**（10 g でも同様）。パックされた先端螺旋（内径 ~4 cm）が 52 mm 球を飲み込めず、横を掠めて押すだけになるのが直接原因です。つまり現キャリブレーション（剛性 0.7、張力 ≤9 N、μ=0.08）に対する物理の答えは「この物体は掴めない」であり、それがこのバックエンドの存在意義です。感度を見るための入口：

```bash
uv run spirob-viewer --grasp                     # 既定 52mm/45g 球
uv run spirob-viewer --grasp --mass 0.01         # 軽くしてみる
uv run spirob-viewer --grasp --size 35           # 小さくしてみる（螺旋内径に合わせる）
uv run spirob-viewer --grasp --stiffness 0.4     # 剛性校正の影響
uv run spirob-viewer --grasp --object soft_sphere --young 8e3   # 柔らかい対象
```

テスト（`test_paper_sequence_reaches_object`）は「安定に完走し、対象に接触する」ことまでを検証し、捕獲は要求していません。

## WebSocket プロトコル

`ws://host:port/ws` に接続すると `{"type": "meta", ...}`（ユニット寸法、flex 頂点数等）が届き、以後 60 Hz で `{"type": "state", ...}`（各ユニットの pos/quat、対象の pos/quat または flex 頂点座標、ケーブル張力、接触ユニット、把持フェーズ）が流れます。クライアントからは `cableForces` / `autoGrasp` / `reset` / `object` コマンドを JSON で送信します。詳細は `server.py` の docstring を参照。
