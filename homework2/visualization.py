# visualization.py
import matplotlib.pyplot as plt
import seaborn as sns

class DataVisualizer:
    """
    Класс для создания и управления визуализациями.
    """
    def __init__(self, rows=1, cols=1, figsize=(12, 8)):
        self.rows = rows
        self.cols = cols
        self.fig, self.axes = plt.subplots(rows, cols, figsize=figsize)
        # Приводим axes к двумерному массиву для удобства
        if rows == 1 and cols == 1:
            self.axes = np.array([[self.axes]])
        elif rows == 1 or cols == 1:
            self.axes = self.axes.reshape(-1, 1) if cols == 1 else self.axes.reshape(1, -1)
        self.current_ax = (0, 0)  # текущая позиция для добавления графика

    def add_histogram(self, data, column, bins=30, ax_pos=None, **kwargs):
        """
        Добавляет гистограмму на указанную позицию (ax_pos) или текущую.
        """
        ax = self._get_ax(ax_pos)
        ax.hist(data[column].dropna(), bins=bins, **kwargs)
        ax.set_title(f'Гистограмма: {column}')
        ax.set_xlabel(column)
        ax.set_ylabel('Частота')
        self._move_current_ax()

    def add_line_plot(self, data, x, y, ax_pos=None, **kwargs):
        """
        Добавляет линейный график.
        """
        ax = self._get_ax(ax_pos)
        ax.plot(data[x], data[y], **kwargs)
        ax.set_title(f'Линейный график: {y} от {x}')
        ax.set_xlabel(x)
        ax.set_ylabel(y)
        self._move_current_ax()

    def add_scatter_plot(self, data, x, y, ax_pos=None, **kwargs):
        """
        Добавляет диаграмму рассеяния.
        """
        ax = self._get_ax(ax_pos)
        ax.scatter(data[x], data[y], **kwargs)
        ax.set_title(f'Диаграмма рассеяния: {x} vs {y}')
        ax.set_xlabel(x)
        ax.set_ylabel(y)
        self._move_current_ax()

    def clear_plots(self, ax_pos=None):
        """
        Удаляет все графики с указанной позиции или полностью очищает полотно.
        Если ax_pos=None, очищается вся фигура.
        """
        if ax_pos is None:
            for i in range(self.rows):
                for j in range(self.cols):
                    self.axes[i, j].clear()
            self.current_ax = (0, 0)
        else:
            ax = self._get_ax(ax_pos)
            ax.clear()

    def show(self):
        plt.tight_layout()
        plt.show()

    def _get_ax(self, ax_pos):
        if ax_pos is None:
            i, j = self.current_ax
        else:
            i, j = ax_pos
        return self.axes[i, j]

    def _move_current_ax(self):
        # Перемещаем текущую позицию на следующую ячейку
        j = self.current_ax[1] + 1
        i = self.current_ax[0]
        if j >= self.cols:
            j = 0
            i += 1
        if i >= self.rows:
            i = 0  # или оставить на последней
        self.current_ax = (i, j)