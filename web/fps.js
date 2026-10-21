// Счётчик кадров (C12): по клавише F показывает в углу частоту кадров и самый долгий кадр за последнюю секунду.
// Критерий плана: при вращении не ниже 50 кадров в секунду и нет рывков.
const WINDOW_MS = 1000;
const GOOD_FPS = 50;
const JERK_MS = 33; // кадр дольше этого — заметный рывок (две «пропущенные» отрисовки при 60 Гц)

export function createFpsMeter(element) {
  const times = [];
  let shown = false;
  let lastText = 0;

  return {
    toggle() {
      shown = !shown;
      element.hidden = !shown;
    },
    /** Вызывается каждый кадр с отметкой времени `performance.now()`. */
    tick(now) {
      times.push(now);
      while (times.length > 1 && times[0] < now - WINDOW_MS) times.shift();
      if (!shown || now - lastText < 250) return;
      lastText = now;
      let worst = 0;
      for (let k = 1; k < times.length; k++) worst = Math.max(worst, times[k] - times[k - 1]);
      const span = times[times.length - 1] - times[0];
      const fps = span > 0 ? ((times.length - 1) * 1000) / span : 0;
      element.textContent = `${Math.round(fps)} кадров/с · худший кадр ${Math.round(worst)} мс`;
      element.classList.toggle("slow", fps < GOOD_FPS || worst > JERK_MS);
    },
  };
}
