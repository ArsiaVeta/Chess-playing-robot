# Chess-playing-robot

## Шахматная сессия

Для установки зависимостей в существующее виртуальное окружение:

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m pytest -q
```

Пример использования:

```python
from src.chess_core.game import ChessGame

game = ChessGame()
game.push("e2e4")
print(game.board)
print(game.outcome)  # None, пока партия продолжается
```

Возможны загрузка позиции через `ChessGame(fen)`, отмена через `undo()` и сдача через `resign()`. Автоматические окончания партии определяются python-chess; заявление ничьей по требованию пока не реализовано.
