Set up a TypeScript project and implement the TaskBoard module per `docs/spec.md`.

**Work directly in the current directory (the workspace root). Do NOT create a project
subfolder** — `package.json`, `tsconfig.json`, and `src/board.ts` must live at the root (the
gates run there and import `./src/board`).

- `npm init -y` in the current directory; install dev deps: `npm install -D typescript vitest`.
- Add `tsconfig.json`: `"target": "ES2020"`, `"module": "ESNext"`, `"moduleResolution": "Bundler"`, `"strict": true`, and `include` `src` and your tests. (Bundler resolution matches vitest/esbuild and allows extensionless relative imports, so `npx tsc --noEmit` and `npx vitest run` agree.)
- Add a `"test": "vitest run"` script to `package.json`.
- Implement the FULL pinned API in `src/board.ts` (exact export names/signatures from `docs/spec.md`):
  `Priority`, `Task`, `Board`, `createBoard`, `addTask`, `completeTask`, `listTasks`, `stats`.
- `addTask` must default priority to "medium", reject empty/blank titles (throw), assign a unique
  id and a strictly-increasing `createdAt`. `listTasks` sorts high>medium>low then createdAt asc.

Make sure `npx tsc --noEmit` is clean before finishing.
