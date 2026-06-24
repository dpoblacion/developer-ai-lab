Write unit tests for the TaskBoard module under `test/` (vitest) and make them pass.

- Cover: addTask returns unique ids + default priority + rejects empty titles; completeTask
  toggles done and throws on unknown id; listTasks filters by done/priority and sorts
  high>medium>low then by createdAt; stats counts total/done/pending.
- Run `npm test` (or `npx vitest run`) and iterate until all your tests pass.
