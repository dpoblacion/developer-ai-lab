// Hidden acceptance suite — copied into the workspace by the `acceptance` gate and run with the
// agent's vitest. Imports the agent's module at ./src/board. The agent never sees this file.
import { describe, it, expect } from "vitest";
import { createBoard, addTask, completeTask, listTasks, stats } from "./src/board";

describe("TaskBoard acceptance", () => {
  it("addTask returns unique ids and defaults priority to medium", () => {
    const b = createBoard();
    const a = addTask(b, { title: "a" });
    const c = addTask(b, { title: "c" });
    expect(a).not.toEqual(c);
    expect(listTasks(b).every((t) => t.priority === "medium")).toBe(true);
  });

  it("addTask rejects empty/blank titles", () => {
    const b = createBoard();
    expect(() => addTask(b, { title: "" })).toThrow();
    expect(() => addTask(b, { title: "   " })).toThrow();
  });

  it("completeTask marks done and throws on unknown id", () => {
    const b = createBoard();
    const id = addTask(b, { title: "x" });
    completeTask(b, id);
    expect(listTasks(b, { done: true }).map((t) => t.id)).toEqual([id]);
    expect(() => completeTask(b, "nope")).toThrow();
  });

  it("listTasks filters by done and by priority", () => {
    const b = createBoard();
    const hi = addTask(b, { title: "hi", priority: "high" });
    addTask(b, { title: "lo", priority: "low" });
    completeTask(b, hi);
    expect(listTasks(b, { done: false }).length).toBe(1);
    expect(listTasks(b, { priority: "high" }).map((t) => t.id)).toEqual([hi]);
  });

  it("listTasks sorts by priority then createdAt ascending", () => {
    const b = createBoard();
    const m = addTask(b, { title: "m", priority: "medium" });
    const h1 = addTask(b, { title: "h1", priority: "high" });
    const h2 = addTask(b, { title: "h2", priority: "high" });
    const l = addTask(b, { title: "l", priority: "low" });
    expect(listTasks(b).map((t) => t.id)).toEqual([h1, h2, m, l]);
  });

  it("stats counts total, done, pending", () => {
    const b = createBoard();
    const a = addTask(b, { title: "a" });
    addTask(b, { title: "b" });
    completeTask(b, a);
    expect(stats(b)).toEqual({ total: 2, done: 1, pending: 1 });
  });
});
