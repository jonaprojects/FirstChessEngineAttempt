import pygame
import pygame_gui
from typing import Tuple, List, Callable, Iterable, Dict, Sized, Optional, Union
from abc import ABC, abstractmethod
import pygame.freetype
from dataclasses import dataclass
from enum import Enum
from pygame import Surface
from pygame.surface import SurfaceType


class PIECE_COLOR(Enum):
    WHITE = 0
    BLACK = 1


pygame.init()
myfont = pygame.font.SysFont("Arial", 80)
RUNNING = True

WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
BROWN = (134, 89, 45)
AQUA = (0, 255, 255)
HIGHLIGHTED_SQUARE_COLOR = (255, 102, 102)
CHOSEN_SQUARE_COLOR = (255, 214, 51)
# CHESS PIECES
# KINGS
BLACK_KING = pygame.image.load("black_king.png")
BLACK_KING = pygame.transform.scale(BLACK_KING, (80, 80))

WHITE_KING = pygame.image.load("white_king.png")
WHITE_KING = pygame.transform.scale(WHITE_KING, (80, 80))

# PAWNS
WHITE_PAWN = pygame.image.load("white_pawn.png")
WHITE_PAWN = pygame.transform.scale(WHITE_PAWN,
                                    (int(WHITE_PAWN.get_width() * 0.25), int(WHITE_PAWN.get_height() * 0.25)))
BLACK_PAWN = pygame.image.load("black_pawn.png")
BLACK_PAWN = pygame.transform.scale(BLACK_PAWN,
                                    (int(BLACK_PAWN.get_width() * 0.25), int(BLACK_PAWN.get_height() * 0.25)))

# ROOKS
WHITE_ROOK = pygame.image.load("white_rook.png")
WHITE_ROOK = pygame.transform.scale(WHITE_ROOK,
                                    (int(WHITE_ROOK.get_width() * 0.25), int(WHITE_ROOK.get_height() * 0.25)))
BLACK_ROOK = pygame.image.load("black_rook.png")
BLACK_ROOK = pygame.transform.scale(BLACK_ROOK,
                                    (int(BLACK_ROOK.get_width() * 0.25), int(BLACK_ROOK.get_height() * 0.25)))
# KNIGHTS

WHITE_KNIGHT = pygame.image.load("white_knight.png")
WHITE_KNIGHT = pygame.transform.scale(WHITE_KNIGHT,
                                      (int(WHITE_KNIGHT.get_width() * 0.25), int(WHITE_KNIGHT.get_height() * 0.25)))
BLACK_KNIGHT = pygame.image.load("black_knight.png")
BLACK_KNIGHT = pygame.transform.scale(BLACK_KNIGHT,
                                      (int(BLACK_KNIGHT.get_width() * 0.25), int(BLACK_KNIGHT.get_height() * 0.25)))
# BISHOPS

WHITE_BISHOP = pygame.image.load("white_bishop.png")
WHITE_BISHOP = pygame.transform.scale(WHITE_BISHOP,
                                      (int(WHITE_BISHOP.get_width() * 0.25), int(WHITE_BISHOP.get_height() * 0.25)))
BLACK_BISHOP = pygame.image.load("black_bishop.png")
BLACK_BISHOP = pygame.transform.scale(BLACK_BISHOP,
                                      (int(BLACK_BISHOP.get_width() * 0.25), int(BLACK_BISHOP.get_height() * 0.25)))

WHITE_QUEEN = pygame.image.load("white_queen.png")
WHITE_QUEEN = pygame.transform.scale(WHITE_QUEEN,
                                     (int(WHITE_QUEEN.get_width() * 0.25), int(WHITE_QUEEN.get_height() * 0.25)))

BLACK_QUEEN = pygame.image.load("black_queen.png")
BLACK_QUEEN = pygame.transform.scale(BLACK_QUEEN,
                                     (int(BLACK_QUEEN.get_width() * 0.25), int(BLACK_QUEEN.get_height() * 0.25)))


def point_in_rect(point: Tuple[float, float], rectangle: pygame.Rect) -> bool:
    return rectangle.x <= point[0] <= rectangle.x + rectangle.width and rectangle.y <= point[
        1] <= rectangle.y + rectangle.height


class Page:
    def __init__(self, width: int, height: int, title: Optional[str] = "My Page",
                 background_color: Tuple[int, int, int] = (230, 230, 230)):
        self._width = width
        self._height = height
        self._title = title
        self._background_color = background_color
        self._window = pygame.display.set_mode((width, height))
        self._window.fill(background_color)
        pygame.display.update()

    @property
    def window(self):
        return self._window

    @property
    def width(self):
        return self._width

    @property
    def height(self):
        return self._height

    @property
    def background(self):
        return self._background_color


class VisualObject(ABC):

    @property
    @abstractmethod
    def x(self):
        pass

    @property
    @abstractmethod
    def y(self):
        pass

    @property
    @abstractmethod
    def screen(self):
        pass

    @abstractmethod
    def draw(self):
        pass


@dataclass
class Text(VisualObject):
    _screen: pygame.Surface
    _x: float
    _y: float
    _text: Optional[str]

    @property
    def x(self):
        return self._x

    @property
    def y(self):
        return self._y

    @property
    def screen(self):
        return self._screen

    def draw(self):
        text_surface = myfont.render('Some Text', False, (0, 0, 0))
        self._screen.blit(text_surface, (self._x, self._y))
        pygame.display.update()


class Square(VisualObject):

    def __init__(self, screen: pygame.Surface, color: Tuple[int, int, int],
                 rectangle: pygame.Rect, current_piece: "Piece" = None, board: "Board" = None,
                 row_index: int = None, column_index: int = None):
        self._screen = screen
        self._rectangle = rectangle
        self._current_piece = current_piece
        self._original_color = color
        self._color = color
        self._board = board
        self._row_index = row_index
        self._column_index = column_index

    @property
    def screen(self):
        return self._screen

    @property
    def x(self):
        return self._rectangle.x

    @property
    def x_middle(self):
        return (self._rectangle.x + self._rectangle.width) / 2

    @property
    def y_middle(self):
        return (self._rectangle.y + self._rectangle.height) / 2

    @property
    def y(self):
        return self._rectangle.y

    @property
    def width(self):
        return self._rectangle.width

    @property
    def height(self):
        return self._rectangle.height

    @property
    def rectangle(self):
        return self._rectangle

    @property
    def is_free(self):
        return self._current_piece is None

    @property
    def color(self):
        return self._color

    @property
    def original_color(self):
        return self._original_color

    def restore_color(self):
        self._color = self._original_color
        self.draw()

    def draw(self):
        pygame.draw.rect(self._screen, self._color, self._rectangle)
        if self._current_piece is not None:
            self._current_piece.draw()
        pygame.display.update()

    def occupy(self, chess_piece: "Piece"):
        self._current_piece = chess_piece

    def free(self):
        self._current_piece = None

    def change_color(self, color: Tuple[int, int, int]):
        self._color = color
        self.draw()

    def on_right_click(self):
        self.change_color(HIGHLIGHTED_SQUARE_COLOR)

    def on_left_click(self):
        self.change_color(CHOSEN_SQUARE_COLOR)
        if not self.is_free:
            for possible_square in self._current_piece.possible_squares():
                possible_square.change_color(CHOSEN_SQUARE_COLOR)

        else:
            print("The square is free")

    @property
    def right_square(self) -> "Optional[Square]":
        return self._board.squares_matrix[self._row_index][self._column_index + 1]

    @property
    def left_square(self) -> "Optional[Square]":
        return self._board.squares_matrix[self._row_index][self._column_index - 1]

    @property
    def up_square(self) -> "Optional[Square]":
        return self._board.squares_matrix[self._row_index - 1][self._column_index]

    @property
    def down_square(self) -> "Optional[Square]":
        return self._board.squares_matrix[self._row_index + 1][self._column_index]

    def has_right(self) -> bool:
        return self._column_index < self._board.num_of_columns - 1

    def right_free(self) -> bool:
        return self.has_right() and self._board.squares_matrix[self._row_index][self._column_index + 1].is_free

    def has_left(self) -> bool:
        return self._column_index > 0

    def left_free(self) -> bool:
        return self.has_left() and self._board.squares_matrix[self._row_index][self._column_index - 1].is_free

    def has_up(self) -> bool:
        return self._row_index > 0

    def up_free(self) -> bool:
        return self.has_up() and self._board.squares_matrix[self._row_index - 1][self._column_index].is_free

    def has_down(self) -> bool:
        return self._row_index < self._board.num_of_rows - 1

    def down_free(self) -> bool:
        return (self.has_down()) and self._board.squares_matrix[self._row_index + 1][self._column_index].is_free

    # SINGLE DIAGONAL SQUARES
    @property
    def diagonal_up_left(self)->"Square":
        return self._board.squares_matrix[self._row_index - 1][
            self._column_index - 1]

    @property
    def diagonal_up_right(self)->"Square":
        return self._board.squares_matrix[self._row_index - 1][
            self._column_index + 1]

    @property
    def diagonal_down_left(self)->"Square":
        return self._board.squares_matrix[self._row_index + 1][
            self._column_index - 1]

    @property
    def diagonal_down_right(self):
        return self._board.squares_matrix[self._row_index + 1][
            self._column_index + 1]


    def has_diagonal_up_left(self):
        return self.has_up() and self.has_left()

    def diagonal_up_left_free(self):
        return self.has_diagonal_up_left() and self._board.squares_matrix[self._row_index - 1][
            self._column_index - 1].is_free

    def has_diagonal_up_right(self):
        return self.has_up() and self.has_right()

    def diagonal_up_right_free(self):
        return self.has_diagonal_up_right() and self._board.squares_matrix[self._row_index - 1][
            self._column_index + 1].is_free

    def has_diagonal_down_left(self):
        return self.has_down() and self.has_left()

    def diagonal_down_left_free(self):
        return self.has_diagonal_down_left() and self._board.squares_matrix[self._row_index + 1][
            self._column_index - 1].is_free

    def has_diagonal_down_right(self):
        return self.has_down() and self.has_right()

    def diagonal_down_right_free(self):
        return self.has_diagonal_down_right() and self._board.squares_matrix[self._row_index + 1][
            self._column_index + 1].is_free

    #  FREE DIAGONALS

    def full_diagonal_up_left(self):
        free_squares = []
        if not self.diagonal_up_left_free():
            return []
        current_square = self.diagonal_up_left
        free_squares.append(current_square)
        while current_square is not None and current_square.is_free and current_square.diagonal_up_left_free():
            up_left_square = current_square.diagonal_up_left
            free_squares.append(up_left_square)
            current_square = up_left_square
        return free_squares

    def full_diagonal_up_right(self):
        if not self.diagonal_up_right_free():
            return []
        free_squares = []
        current_square = self.diagonal_up_right
        free_squares.append(current_square)
        while current_square is not None and current_square.is_free and current_square.diagonal_up_right_free():
            up_right_square = current_square.diagonal_up_right
            free_squares.append(up_right_square)
            current_square = up_right_square
        return free_squares

    def full_diagonal_down_left(self):
        if not self.diagonal_down_left_free():
            return []
        free_squares = []
        current_square = self.diagonal_down_left
        free_squares.append(current_square)
        while current_square is not None and current_square.is_free and current_square.diagonal_down_left_free():
            down_left_square = current_square.diagonal_down_left
            free_squares.append(down_left_square)
            current_square = down_left_square
        return free_squares

    def full_diagonal_down_right(self):
        if not self.diagonal_down_right_free():
            return []
        free_squares = []
        current_square = self.diagonal_down_right
        free_squares.append(current_square)
        while current_square is not None and current_square.is_free and current_square.diagonal_down_right_free():
            down_right_square = current_square.diagonal_down_right
            free_squares.append(down_right_square)
            current_square = down_right_square
        return free_squares

    def full_free_diagonals(self):
        return self.full_diagonal_up_left() + self.full_diagonal_up_right() + self.full_diagonal_down_left() + self.full_diagonal_down_right()

    # FREE ROWS AND COLUMNS

    def free_squares_in_row(self, row_index: int):
        free_squares: List[Square] = list()
        for column_index in range(self._column_index - 1, -1, -1):
            # Stop checking when the row is blocked, because it is not accessible from there
            square = self._board.squares_matrix[row_index][column_index]
            if not square.is_free:
                break
            free_squares.append(square)
        for column_index in range(self._column_index + 1, self._board.num_of_columns, 1):
            square = self._board.squares_matrix[row_index][column_index]
            if not square.is_free:  # Stop checking when the row is blocked, because it is not accessible from there
                break
        return free_squares

    def free_squares_in_column(self, column_index: int):
        free_squares = []
        for row_index in range(self._row_index - 1, -1, -1):
            square = self._board.squares_matrix[row_index][column_index]
            if not square.is_free:
                break
            free_squares.append(square)

        for row_index in range(self._row_index + 1, self._board.num_of_rows, 1):
            square = self._board.squares_matrix[row_index][column_index]
            if not square.is_free:
                break
            free_squares.append(square)

        return free_squares

    @property
    def real_row(self):
        return self._board.num_of_rows - self._row_index

    @property
    def column_letter(self):
        return chr(ord('a') + self._column_index)


class Board:
    def __init__(self, screen: pygame.Surface, rectangle: pygame.Rect, row_squares: int = 8, column_squares: int = 8,
                 colors: Tuple[Tuple[int, int, int], Tuple[int, int, int]] = None):
        self._screen = screen
        if colors is None:
            self._colors = (WHITE, BLACK)
        else:
            self._colors = colors
        self._rectangle = rectangle
        if self._rectangle.width > self._screen.get_width() or self._rectangle.height > self._screen.get_height():
            raise ValueError("The main_board exceeds the screen's size")
        self._squares: List[List[Optional[Square]]] = [[None for j in range(column_squares)] for i in
                                                       range(row_squares)]
        self._square_width, self._square_height = self._rectangle.width / 8, self._rectangle.height / 8

        self.generate_board()

    def generate_board(self):
        for row_index, row in enumerate(self._squares):
            for column_index, column in enumerate(row):
                self._squares[row_index][column_index] = Square(
                    screen=self._screen,
                    color=self._colors[(row_index + column_index) % 2],
                    rectangle=pygame.Rect(
                        self._rectangle.x + self._square_width * column_index,
                        self._rectangle.y + self._square_height * row_index,
                        self._square_width,
                        self._square_height
                    ),
                    board=self,
                    row_index=row_index,
                    column_index=column_index
                )

    @property
    def screen(self):
        return self._screen

    @property
    def width(self):
        return self._rectangle.width

    @property
    def height(self):
        return self._rectangle.height

    @property
    def num_of_rows(self):
        return len(self._squares)

    @property
    def num_of_columns(self):
        return len(self._squares[0])

    @property
    def squares_matrix(self):
        return self._squares

    @property
    def rectangle(self):
        return self._rectangle

    def draw(self):
        for row in self._squares:
            for square in row:
                square.draw()

    def restore_all_colors(self):
        for row in self._squares:
            for square in row:
                square.restore_color()

    def valid_row_index(self, row_index: int) -> bool:
        return 0 <= row_index < self.num_of_rows

    def valid_column_index(self, column_index: int) -> bool:
        return 0 <= column_index < self.num_of_columns


class Button:
    def __init__(self, screen: Union[Surface, pygame.Surface], rectangle: pygame.Rect, text: str = "",
                 on_click: Callable = None):
        self._screen = screen
        self._rectangle = rectangle
        self._text = text


class Piece(VisualObject):
    def __init__(self, board: Board, piece_color: PIECE_COLOR,
                 value: int, location: Tuple[int, int], image: Union[Surface, pygame.Surface]):
        self._board = board
        self._color = piece_color
        self._value = value
        self._row, self._column = location
        self._square: Square = board.squares_matrix[self._row][self._column]
        self._square.occupy(self)
        self._image = image

    @property
    def screen(self):
        return self._board.screen

    @property
    def square(self):
        return self._square

    @square.setter
    def square(self, new_square: Square):
        self._square = new_square

    @property
    def x(self):
        return self._square.x

    @property
    def y(self):
        return self._square.y

    @property
    def color(self):
        return self._color

    @property
    def current_row(self):
        return self._row

    @property
    def current_column(self):
        return self._column

    def draw(self):
        x = self._square.x + (self._square.width - self._image.get_rect().width) // 2
        y = self._square.y + (self._square.height - self._image.get_rect().height) // 2

        self._board.screen.blit(self._image, (x, y))
        pygame.display.update()

    def possible_squares(self):
        pass

    def move(self):
        pass


class King(Piece):
    def __init__(self, board: Board, location: Tuple[int, int], piece_color: PIECE_COLOR):
        super().__init__(
            board=board,
            location=location,
            piece_color=piece_color,
            value=10,
            image=BLACK_KING if piece_color.value else WHITE_KING
        )
        self._has_moved = False  # When it moves, it cannot castle !

    def possible_squares(self):
        possible_squares = []
        # CHECKING SIDES
        if self._square.right_free():
            possible_squares.append(self._square.right_square)
        if self._square.left_free():
            possible_squares.append(self._square.left_square)
        if self._square.up_free():
            possible_squares.append(self._square.up_square)
        if self._square.down_free():
            possible_squares.append(self._square.down_square)

        # CHECKING DIAGONALS
        if self._square.diagonal_up_right_free():
            possible_squares.append(self._square.diagonal_up_right)
        if self._square.diagonal_down_right_free():
            possible_squares.append(self._square.diagonal_down_right)
        if self._square.diagonal_up_left_free():
            possible_squares.append(self._square.diagonal_up_left)
        if self._square.diagonal_down_right_free():
            possible_squares.append(self._square.diagonal_up_left)
        return possible_squares

    def can_castle(self):  # TODO: CHECK WITH THE ROOKS, AND IF THE SQUARES IN BETWEEN ARE EMPTY
        if not self._has_moved:
            if self._color == PIECE_COLOR.WHITE:
                pass
            else:
                pass


class Queen(Piece):
    def __init__(self, board: Board, location: Tuple[int, int], piece_color: PIECE_COLOR):
        super().__init__(
            board=board,
            location=location,
            piece_color=piece_color,
            value=9,
            image=BLACK_QUEEN if piece_color.value else WHITE_QUEEN
        )

    def possible_squares(self):
        return self._square.free_squares_in_row(self._row) + self._square.free_squares_in_column(self._column) + self._square.full_free_diagonals()


class Rook(Piece):
    def __init__(self, board: Board, location: Tuple[int, int], piece_color: PIECE_COLOR):
        super().__init__(
            board=board,
            location=location,
            piece_color=piece_color,
            value=5,
            image=BLACK_ROOK if piece_color.value else WHITE_ROOK
        )

    def possible_squares(self):
        return self._square.free_squares_in_row(self._row) + self._square.free_squares_in_column(self._column)


class Bishop(Piece):
    def __init__(self, board: Board, location: Tuple[int, int], piece_color: PIECE_COLOR):
        super().__init__(
            board=board,
            location=location,
            piece_color=piece_color,
            value=4,
            image=BLACK_BISHOP if piece_color.value else WHITE_BISHOP
        )

    def possible_squares(self):
        return self._square.full_free_diagonals()


class Knight(Piece):
    def __init__(self, board: Board, location: Tuple[int, int], piece_color: PIECE_COLOR):
        super().__init__(
            board=board,
            location=location,
            piece_color=piece_color,
            value=3,
            image=BLACK_KNIGHT if piece_color.value else WHITE_KNIGHT
        )

    def possible_squares(self):
        possible_positions = []
        possible_indices = (
            (self._row - 2, self._column + 1),
            (self._row - 2, self._column - 1),
            (self._row - 1, self._column + 2),
            (self._row - 1, self._column - 2),
            (self._row + 2, self._column + 1),
            (self._row + 2, self._column - 1),
            (self._row + 1, self._column + 2),
            (self._row + 1, self._column - 2),

        )
        for row, column in possible_indices:
            if self._board.valid_row_index(row) and self._board.valid_column_index(column) and (
                    square := self._board.squares_matrix[row][column]).is_free:
                possible_positions.append(square)

        return possible_positions


class Pawn(Piece):
    def __init__(self, board: Board, location: Tuple[int, int], piece_color: PIECE_COLOR):
        super().__init__(
            board=board,
            location=location,
            piece_color=piece_color,
            value=1,
            image=BLACK_PAWN if piece_color.value else WHITE_PAWN
        )

    def possible_squares(self):
        possible_squares: List[Optional[Square]] = []
        if self._color == PIECE_COLOR.WHITE:
            if self._square.up_free():  # TODO: later add a check whether it's a legal move !
                possible_squares.append(self._square.up_square)
                if self._row == 6 and self._square.up_square.up_free():  # The pawn hasn't moved yet
                    possible_squares.append(self._square.up_square.up_square)
        else:
            if self._square.down_free():
                possible_squares.append(self._square.down_square)
                if self._row == 1 and self._square.down_square.down_free():  # The pawn hasn't moved yet
                    possible_squares.append(self._square.down_square.down_square)

        return possible_squares


main_page = Page(1100, 900, "Chess Engine V1.0")
main_board = Board(main_page.window, pygame.Rect(165, 60, 780, 780), colors=(WHITE, BROWN))

black_king = King(main_board, (0, 4), PIECE_COLOR.BLACK)
white_king = King(main_board, (7, 4), PIECE_COLOR.WHITE)

white_queen = Queen(main_board, (7, 3), PIECE_COLOR.WHITE)
black_queen = Queen(main_board, (0, 3), PIECE_COLOR.BLACK)

right_white_rook = Rook(main_board, (7, 7), PIECE_COLOR.WHITE)
left_white_rook = Rook(main_board, (7, 0), PIECE_COLOR.WHITE)
right_black_rook = Rook(main_board, (0, 7), PIECE_COLOR.BLACK)
left_black_rook = Rook(main_board, (0, 0), PIECE_COLOR.BLACK)

right_white_knight = Knight(main_board, (7, 6), PIECE_COLOR.WHITE)
left_white_knight = Knight(main_board, (7, 1), PIECE_COLOR.WHITE)
right_black_knight = Knight(main_board, (0, 6), PIECE_COLOR.BLACK)
left_black_knight = Knight(main_board, (0, 1), PIECE_COLOR.BLACK)

right_white_bishop = Bishop(main_board, (7, 5), PIECE_COLOR.WHITE)
left_white_bishop = Bishop(main_board, (7, 2), PIECE_COLOR.WHITE)
right_black_bishop = Bishop(main_board, (0, 5), PIECE_COLOR.BLACK)
left_black_bishop = Bishop(main_board, (0, 2), PIECE_COLOR.BLACK)



for column_index in range(8):
    white_pawn = Pawn(main_board, (1, column_index), PIECE_COLOR.BLACK)
main_board.draw()


def square_clicked(board: Board, coordinate: Tuple[float, float]) -> Optional[Tuple[Square, Tuple[int, int]]]:
    for row_index, row in enumerate(board.squares_matrix):
        for column_index, square in enumerate(row):
            if point_in_rect(coordinate, square.rectangle):
                return square, (row_index, column_index)

    return None  # The player didn't choose any square


def clicked_in_board(board: Board, coordinate: Tuple[float, float]):
    return point_in_rect(coordinate, board.rectangle)


def main():
    global RUNNING
    while RUNNING:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                RUNNING = False
            elif event.type == pygame.MOUSEBUTTONDOWN:
                x, y = pygame.mouse.get_pos()
                if event.button == 1:  # LEFT CLICK
                    if clicked_in_board(main_board, (x, y)):
                        main_board.restore_all_colors()
                        result: Optional[Tuple[Square, Tuple[int, int]]] = square_clicked(main_board, (x, y))
                        if result is not None:
                            if result[0].color == CHOSEN_SQUARE_COLOR:
                                result[0].restore_color()
                            else:
                                result[0].on_left_click()

                elif event.button == 3:  # RIGHT CLICK
                    if clicked_in_board(main_board, (x, y)):
                        result: Optional[Tuple[Square, Tuple[int, int]]] = square_clicked(main_board, (x, y))
                        if result is not None:  # A square was selected with the right click
                            if result[0].color == HIGHLIGHTED_SQUARE_COLOR:
                                result[0].restore_color()
                            else:
                                result[0].on_right_click()


if __name__ == '__main__':
    main()
