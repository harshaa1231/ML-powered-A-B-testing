"use client";

import { useCallback, useState } from "react";

const STORAGE_KEY = "abtesting_progress";

export interface ProgressState {
  completedLessons: string[];
  completedCaseStudies: string[];
  correctQuizzes: string[];
  completedPracticeScenarios: string[];
  xp: number;
  streak: {
    current: number;
    longest: number;
    lastActiveDate: string | null; // YYYY-MM-DD
  };
}

const XP_AWARDS = {
  lesson: 10,
  quiz: 5,
  caseStudy: 15,
  practice: 20,
} as const;

function defaultState(): ProgressState {
  return {
    completedLessons: [],
    completedCaseStudies: [],
    correctQuizzes: [],
    completedPracticeScenarios: [],
    xp: 0,
    streak: { current: 0, longest: 0, lastActiveDate: null },
  };
}

function todayKey(): string {
  return new Date().toISOString().slice(0, 10);
}

function daysBetween(a: string, b: string): number {
  const msPerDay = 1000 * 60 * 60 * 24;
  return Math.round((new Date(b).getTime() - new Date(a).getTime()) / msPerDay);
}

function readState(): ProgressState {
  if (typeof window === "undefined") return defaultState();
  try {
    const raw = window.localStorage.getItem(STORAGE_KEY);
    if (!raw) return defaultState();
    return { ...defaultState(), ...JSON.parse(raw) };
  } catch {
    return defaultState();
  }
}

function writeState(state: ProgressState): void {
  if (typeof window === "undefined") return;
  try {
    window.localStorage.setItem(STORAGE_KEY, JSON.stringify(state));
  } catch {
    // localStorage unavailable (private browsing, quota, etc.) — progress just won't persist.
  }
}

function bumpStreak(state: ProgressState): ProgressState {
  const today = todayKey();
  const { lastActiveDate, current, longest } = state.streak;

  if (lastActiveDate === today) return state; // already counted today

  const gap = lastActiveDate ? daysBetween(lastActiveDate, today) : null;
  const nextCurrent = gap === 1 ? current + 1 : 1; // consecutive day vs. reset
  const nextLongest = Math.max(longest, nextCurrent);

  return { ...state, streak: { current: nextCurrent, longest: nextLongest, lastActiveDate: today } };
}

function addXp(state: ProgressState, amount: number): ProgressState {
  return bumpStreak({ ...state, xp: state.xp + amount });
}

export function useProgress() {
  const [state, setState] = useState<ProgressState>(readState);

  const mutate = useCallback((updater: (prev: ProgressState) => ProgressState) => {
    setState((prev) => {
      const next = updater(prev);
      writeState(next);
      return next;
    });
  }, []);

  const completeLesson = useCallback(
    (lessonId: string) => {
      mutate((prev) => {
        if (prev.completedLessons.includes(lessonId)) return prev;
        return addXp({ ...prev, completedLessons: [...prev.completedLessons, lessonId] }, XP_AWARDS.lesson);
      });
    },
    [mutate]
  );

  const completeCaseStudy = useCallback(
    (caseStudyId: string) => {
      mutate((prev) => {
        if (prev.completedCaseStudies.includes(caseStudyId)) return prev;
        return addXp({ ...prev, completedCaseStudies: [...prev.completedCaseStudies, caseStudyId] }, XP_AWARDS.caseStudy);
      });
    },
    [mutate]
  );

  const recordQuizAnswer = useCallback(
    (quizId: string, correct: boolean) => {
      mutate((prev) => {
        if (!correct || prev.correctQuizzes.includes(quizId)) return prev;
        return addXp({ ...prev, correctQuizzes: [...prev.correctQuizzes, quizId] }, XP_AWARDS.quiz);
      });
    },
    [mutate]
  );

  const completePracticeScenario = useCallback(
    (scenarioKey: string) => {
      mutate((prev) => {
        if (prev.completedPracticeScenarios.includes(scenarioKey)) return prev;
        return addXp(
          { ...prev, completedPracticeScenarios: [...prev.completedPracticeScenarios, scenarioKey] },
          XP_AWARDS.practice
        );
      });
    },
    [mutate]
  );

  return { progress: state, completeLesson, completeCaseStudy, recordQuizAnswer, completePracticeScenario };
}
