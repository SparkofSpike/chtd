# Running against Agnes

The hackathon's models are Agnes ones, so the teacher route points at Agnes. This records
the arrangement that works, the two mistakes made getting there, and what the models can
do.

Status: **an Agnes teacher drives the frozen track end to end, with the key the
organisers issued.**

```
teacher   agnes-3.0-flash          https://api.agnes-ai.cn/v1        (the issued key)
student   qwen3-4b-instruct-2507   http://100.113.249.5:8000/v1      (the training box)
```

The student is local because it is the side that gets LoRA trained, and a hosted API has
no weights to train. The competition's rules permit this explicitly — see *Compliance*
below.

## The key was fine; the reading and the endpoint were not

Two errors, and both had to be fixed before anything worked.

**The endpoint was wrong.** The CodeWhale config carries an Agnes credential at
`apihub.agnes-ai.com/v1`, and that is the host every early attempt was aimed at. The
documentation names a different one:

```
https://api.agnes-ai.cn/v1/chat/completions
```

They are separate platforms with separate key registries. A key valid on one answers 401
on the other, which reads exactly like a dead key.

**One character was misread.** At index 32 the glyph is uppercase `I`:

```
as first transcribed   ...oG57lBDtK4...   401 无效的令牌
as it actually is      ...oG57IBDtK4...   403 该令牌无权访问模型
```

`I`, `l` and `1` are near-identical in the font the email renders in, and this was caught
by reading the key back rather than by anything clever.

**A test flaw hid both.** The first exhaustive attempt tried every combination of the
ambiguous characters — 144 readings — and reported all of them refused. That was wrong.
The script tested `if status == 200` and treated everything else as a refusal, so the
**403 that said the key was valid** was filed in the same bucket as 401s. A correct answer
was produced and discarded by the classification.

## 401 and 403 mean different things

This distinction is worth stating on its own, because conflating them is what cost the
time:

| status | message | meaning |
| --- | --- | --- |
| 401 | `无效的令牌` / `Invalid token` | the token is not known here |
| 401 | `未提供令牌` / `Token not provided` | no credential was sent |
| **403** | **`该令牌无权访问模型 X`** | **the token is valid and X is not granted to it** |
| 503 | `model_not_found` | the model name does not exist at all |

A credential probe must separate these. `agnes_credentials.py` returns one of
`allowed` / `scoped` / `rejected`, where `scoped` means "endpoint and key are right, the
model is wrong" — the state that looks like failure and is not.

## The issued key is scoped to one model

```
agnes-3.0-flash      200  allowed
agnes-2.5-flash      403  not granted
agnes-2.5-pro        403  not granted
agnes-3.0-pro        403  not granted
(and ten other spellings, all 403)
```

So the endpoint and the model travel together with the key, and nothing may hardcode
either. `agnes_credentials.py` resolves the triple; `make_agnes_profile.py` reads it from
there rather than from constants.

## What the model can do

`agnes-3.0-flash`, measured through the issued key:

| capability | result |
| --- | --- |
| tool call when one is the obvious move | `read_file({"path": "config.json"})` |
| refrains when no tool is needed | `2 + 2` answers `4`, no call |
| streaming | 17 chunks, assembled correctly |
| usage accounting | present |
| latency | 1.8–5.8 s for a short turn |

## The run

```
student  session=...6c5e4864   preset=chtd   1 decision
teacher  session=...1339cb90   preset=chtd   2 decisions

shared state 935ef0d446090af60b5748a48b5cb847fa5440ff945ad705ad0e40f3546b0e1f
  student: Hello! I'm ready to assist you. How can I help you today?
  teacher: read({"path":"notes.txt"})

teacher answer 'The word on the second line of notes.txt is **bravo**.'  correct
student answer 'Hello! I'm ready to assist you.'                          incorrect

1 divergence, 0 agreements, 1 novel  ->  admitted: 1, refused: 0
dataset digest a8c373b104ab442c4780c7f3f9914014cafe1243ec5ecfe7123f14211e525f9b
```

## Compliance

The registration carries a technical commitment that models are limited to Agnes, and the
project's student is a Qwen. The competition's own rules resolve this, in 3.2:

> 报名成功后，请加入官方赛事群，赛事组委会将统一向参赛团队提供 **Agnes 自研模型调用 Key** …
> 如参赛团队使用其他开源模型或第三方商业模型，所需的**账号、API Key、调用额度及相关费用由参赛团队自行解决并承担**。

So third-party and open models are permitted, at the team's own cost, provided the
submission states each model's name, version, where it is used and how it is called. The
teacher is an Agnes model — the one the organisers issued — so the core task runs on the
platform the competition names, and the student's use is declared.

What is *not* available is the design's control for model family: CHTD asks the teacher
and student to share a tokenizer, to rule out tokenizer and pretraining differences. That
control is absent here. It is smaller than it sounds — the teacher is reached over HTTP,
so it reads the student's state as text and answers as text, and no logit-level comparison
happens anywhere — but the control is genuinely missing and belongs in the write-up.

## Submission details

From the same source, for reference:

- artifact naming: **组别-参赛编号-队伍名称-队长姓名** (this team: `本科生组-U240-Mojito`)
- required evidence: AGH 执行记录, ≥1 条工具调用链, Agnes 模型参与核心任务证据,
  测试样例 (正常/边界/失败三类)
- demo video 3–5 minutes, **link only**, not uploaded
- review 10-16…10-18; final 10-20
- top 100 per group receive 3 months of Agnes Pro Token Plan from 10-23
- the 20 winning teams open-source their repositories within 5 working days
- excluded: 纯概念 / PPT / 界面原型 / 纯聊天

## Operational notes

**`--preset` on the command line does not work, and the track still does.** Naming the
preset is rejected with `PRESET_SWITCH_REJECTED`; omitting it runs under
`presets.default`, which is the same `cthd`, and the ledger records `preset: chtd`. The
real error is unrecoverable — `mapCore` re-raises any of four preset/model codes as one
RPC error and the display layer drops the reason. Do not pass the flag; read the preset
from the ledger.

**A rate limit shapes the run.** A burst returns 429; one request every six seconds
succeeded ten times out of ten. A rollout needs throttling and retry, not parallelism.

**Keys never enter the repository.** They live in `D:\CodeWhaleData\secrets` and reach the
daemon as `AGNES_SECRET_TEACHER_DEFAULT` / `AGNES_SECRET_STUDENT_DEFAULT`, through the
profile's `secrets: { kind: env }` adapter. `make_agnes_profile.py` refuses to write a
profile containing anything that looks like a credential.
