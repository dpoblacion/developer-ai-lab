You are building a small TypeScript library called **TaskBoard** — a pure, in-memory task
list. This phase: write `docs/spec.md` capturing the design and edge cases. Implement nothing yet.

Implement (in later phases) EXACTLY this public API, in `src/board.ts`:

    export type Priority = "low" | "medium" | "high";
    export interface Task { id: string; title: string; priority: Priority; done: boolean; createdAt: number; }
    export interface Board { /* your choice */ }

    export function createBoard(): Board;
    // addTask: returns a NEW unique id. priority defaults to "medium". Throws on empty/blank title.
    //   createdAt is a STRICTLY INCREASING number (insertion order).
    export function addTask(board: Board, input: { title: string; priority?: Priority }): string;
    // completeTask: sets done=true for that task. Throws if the id is unknown.
    export function completeTask(board: Board, id: string): void;
    // listTasks: optional filter by done and/or priority. Sorted by priority (high>medium>low),
    //   ties broken by createdAt ascending.
    export function listTasks(board: Board, filter?: { done?: boolean; priority?: Priority }): Task[];
    // stats: { total, done, pending }.
    export function stats(board: Board): { total: number; done: number; pending: number };

Write `docs/spec.md`: a short description of each function, the priority sort order, and the
edge cases (empty title, unknown id, default priority, the strictly-increasing createdAt).
