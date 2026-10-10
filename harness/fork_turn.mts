/** Continue one completed student turn from the same durable AGH prefix in two arms. */
import { readFileSync } from 'node:fs'
import { resolve } from 'node:path'
import { pathToFileURL } from 'node:url'

function required(name: string): string {
  const index = process.argv.indexOf(`--${name}`)
  if (index < 0 || !process.argv[index + 1]) throw new Error(`missing --${name}`)
  return process.argv[index + 1]
}

const aghSource = resolve(required('agh-source'))
const home = resolve(required('agh-home'))
const cwd = resolve(required('cwd'))
const parentKey = required('parent')
const at = Number(required('at'))
const prompt = required('prompt')
const key = readFileSync(resolve(required('key-file')), 'utf8').trim()
if (!key.startsWith('sk-')) throw new Error('invalid Agnes credential file')
if (!Number.isSafeInteger(at) || at <= 0) throw new Error('fork boundary must be a positive event sequence')
process.env.AGNES_SECRET_TEACHER_DEFAULT = key
process.env.AGNES_SECRET_STUDENT_DEFAULT = 'local-no-auth'
process.env.AGH_HOME = home
process.env.AGNES_PROFILE = 'local-dev'

const bootFile = pathToFileURL(resolve(aghSource, 'packages/cli/src/boot/local.ts')).href
const { bootLocal } = await import(bootFile)
const parsed = {
  positional: [], rest: [], print: true, help: false, version: false,
  profile: 'local-dev', cwd, continue: false, park: false,
  meta: false, ephemeral: false, standalone: true, json: false,
  repair: false, upgrade: false, html: false, raw: false,
  resolved: false, probe: false,
}
const booted = await bootLocal(parsed, {
  env: process.env, home, cwd, agnesVersion: '0.0.0', log: () => {},
})
try {
  const parent = await booted.client.session.load(parentKey, { cwd })
  const child = await booted.client.session.fork(parentKey, at)
  await child.setModel({ slot: 'primary', route: 'teacher', model: 'agnes-3.0-flash' })
  await child.attach({ filter: { acpUpdates: false } })
  const teacher = await child.prompt(prompt)
  // Run the student only after the fork is durable and the teacher has finished.
  await parent.setModel({ slot: 'primary', route: 'student', model: 'qwen3-4b-instruct-2507' })
  await parent.attach({ filter: { acpUpdates: false } })
  const student = await parent.prompt(prompt)
  process.stdout.write(JSON.stringify({
    parent_key: parentKey, child_key: child.id, boundary_seq: at,
    teacher_reason: teacher.reason, teacher_last_seq: teacher.lastSeq,
    student_reason: student.reason, student_last_seq: student.lastSeq,
  }) + '\n')
} finally {
  await booted.close()
}
