import pygame
from typing import Tuple, List, Callable, Iterable, Dict, Sized, Optional, Union
from abc import ABC, abstractmethod
import pygame.freetype
from dataclasses import dataclass
from enum import Enum
from pygame import Surface


class PIECE_COLOR(Enum):
    WHITE = 0
    BLACK = 1


class PIECES_LETTERS(Enum):
    KING = 'K'
    QUEEN = 'Q'
    ROOK = 'R'
    BISHOP = 'B'
    KNIGHT = 'N'
    PAWN = ''


pygame.init()
myfont = pygame.font.SysFont("Arial", 80)
RUNNING = True
MOVING_PIECE = False  # Move this perhaps to a class in the future
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


def generate_pieces(given_board) -> "List[Piece]":
    pieces: List[Piece] = list()
    pieces.append(King(given_board, (7, 4), PIECE_COLOR.WHITE))  # WHITE KING
    pieces.append(King(given_board, (0, 4), PIECE_COLOR.BLACK))  # BLACK KING

    pieces.append(Queen(given_board, (7, 3), PIECE_COLOR.WHITE))  # WHITE QUEEN
    pieces.append(Queen(given_board, (0, 3), PIECE_COLOR.BLACK))  # BLACK QUEEN

    pieces.append(Rook(given_board, (7, 7), PIECE_COLOR.WHITE))  # RIGHT WHITE ROOK
    pieces.append(Rook(given_board, (7, 0), PIECE_COLOR.WHITE))  # LEFT WHITE ROOK
    pieces.append(Rook(given_board, (0, 7), PIECE_COLOR.BLACK))  # RIGHT BLACK ROOK
    pieces.append(Rook(given_board, (0, 0), PIECE_COLOR.BLACK))  # LEFT BLACK ROOK

    pieces.append(Knight(given_board, (7, 6), PIECE_COLOR.WHITE))  # RIGHT WHITE KNIGHT
    pieces.append(Knight(given_board, (7, 1), PIECE_COLOR.WHITE))  # LEFT WHITE KNIGHT
    pieces.append(Knight(given_board, (0, 6), PIECE_COLOR.BLACK))  # RIGHT BLACK KNIGHT
    pieces.append(Knight(given_board, (0, 1), PIECE_COLOR.BLACK))  # LEFT BLACK KNIGHT

    pieces.append(Bishop(given_board, (7, 5), PIECE_COLOR.WHITE))  # RIGHT WHITE BISHOP
    pieces.append(Bishop(given_board, (7, 2), PIECE_COLOR.WHITE))  # LEFT WHITE BISHOP
    pieces.append(Bishop(given_board, (0, 5), PIECE_COLOR.BLACK))  # RIGHT BLACK BISHOP
    pieces.append(Bishop(given_board, (0, 2), PIECE_COLOR.BLACK))  # LEFT BLACK BISHOP

    for column_index in range(8):  # Adding white pawns
        pieces.append(Pawn(given_board, (6, column_index), PIECE_COLOR.WHITE))

    for column_index in range(8):  # Adding black pawns
        pieces.append(Pawn(given_board, (1, column_index), PIECE_COLOR.BLACK))

    return pieces


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
        pygame.display.set_caption(self._title)
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

    @property
    def row_index(self):
        return self._row_index

    @property
    def column_index(self):
        return self._column_index

    @property
    def current_piece(self):
        return self._current_piece

    def restore_color(self):
        self._color = self._original_color
        self.draw()

    def draw(self):
        pygame.draw.rect(self._screen, self._color, self._rectangle)
        if self._current_piece is not None:
            self._current_piece.draw()

        pygame.display.update()

    def occupy(self, chess_piece: "Piece"):
        chess_piece.row, chess_piece.column = self._row_index, self._column_index
        chess_piece.square = self
        self._current_piece = chess_piece
        self.draw()

    def free(self):
        self._current_piece = None
        self.draw()

    def change_color(self, color: Tuple[int, int, int]):
        self._color = color
        self.draw()

    def on_right_click(self):
        self.change_color(HIGHLIGHTED_SQUARE_COLOR)

    def on_left_click(self):
        print('pressed left click !')
        self.change_color(CHOSEN_SQUARE_COLOR)
        if not self.is_free:
            for possible_square in self._current_piece.possible_squares():
                possible_square.change_color(CHOSEN_SQUARE_COLOR)
            if game_handler.current_move.chosen_piece is None:
                print("chose a piece !")
                game_handler.set_chosen_piece(self._current_piece)
            else:
                game_handler.set_destination_square(self)
                game_handler.execute_move()  # Won't execute if it's not valid!
        elif game_handler.current_move.chosen_piece is not None:
            print("moving the piece !")
            game_handler.set_destination_square(self)
            game_handler.execute_move()
        else:
            print("Clear everything !")

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
    def diagonal_up_left(self) -> "Optional[Square]":
        if not self.has_diagonal_up_left():
            return None
        return self._board.squares_matrix[self._row_index - 1][
            self._column_index - 1]

    @property
    def diagonal_up_right(self) -> "Optional[Square]":
        if not self.has_diagonal_up_right():
            return None
        return self._board.squares_matrix[self._row_index - 1][
            self._column_index + 1]

    @property
    def diagonal_down_left(self) -> "Optional[Square]":
        if not self.has_diagonal_down_left():
            return None
        return self._board.squares_matrix[self._row_index + 1][
            self._column_index - 1]

    @property
    def diagonal_down_right(self) -> "Optional[Square]":
        if not self.has_diagonal_down_right():
            return None
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
        if not self.has_diagonal_up_left():
            return []
        current_square = self.diagonal_up_left
        free_squares.append(current_square)
        while current_square is not None and current_square.is_free:
            free_squares.append(current_square)
            current_square = current_square.diagonal_up_left
        if current_square is not None:  # Diagonal getting blocked
            if self._current_piece.can_capture(current_square._current_piece):
                free_squares.append(current_square)
        return free_squares

    def full_diagonal_up_right(self):
        if not self.has_diagonal_up_right():
            return []
        free_squares = []
        current_square = self.diagonal_up_right
        free_squares.append(current_square)
        while current_square is not None and current_square.is_free:
            free_squares.append(current_square)
            current_square = current_square.diagonal_up_right
        if current_square is not None:  # Diagonal getting blocked
            if self._current_piece.can_capture(current_square._current_piece):
                free_squares.append(current_square)
        return free_squares

    def full_diagonal_down_left(self):
        if not self.has_diagonal_down_left():
            return []
        free_squares = []
        current_square = self.diagonal_down_left
        free_squares.append(current_square)
        while current_square is not None and current_square.is_free:
            free_squares.append(current_square)
            current_square = current_square.diagonal_down_left
        if current_square is not None:  # Diagonal getting blocked
            if self._current_piece.can_capture(current_square._current_piece):
                free_squares.append(current_square)
        return free_squares

    def full_diagonal_down_right(self):
        if not self.has_diagonal_down_right():
            return []
        free_squares = []
        current_square = self.diagonal_down_right
        free_squares.append(current_square)
        while current_square is not None and current_square.is_free:
            free_squares.append(current_square)
            current_square = current_square.diagonal_down_right
        if current_square is not None:  # Diagonal getting blocked
            if self._current_piece.can_capture(current_square._current_piece):
                free_squares.append(current_square)
        return free_squares

    def full_free_diagonals(self):
        return self.full_diagonal_up_left() + self.full_diagonal_up_right() + self.full_diagonal_down_left() + self.full_diagonal_down_right()

    # FREE ROWS AND COLUMNS

    def free_squares_in_row(self, row_index: int):
        free_squares: List[Square] = list()
        for column_index in range(self._column_index - 1, -1, -1):
            # Stop checking when the row is blocked, because it is not accessible from there
            square = self._board.squares_matrix[row_index][column_index]
            if square.is_free:
                free_squares.append(square)
            elif not square.is_free and self._current_piece.can_capture(square._current_piece):
                free_squares.append(square)
                break
            else:
                break
        for column_index in range(self._column_index + 1, self._board.num_of_columns, 1):
            square = self._board.squares_matrix[row_index][column_index]
            if square.is_free:
                free_squares.append(square)
            elif not square.is_free and self._current_piece.can_capture(square._current_piece):
                free_squares.append(square)
                break
            else:
                break

        return free_squares

    def free_squares_in_column(self, column_index: int):
        free_squares = []
        for row_index in range(self._row_index - 1, -1, -1):
            square = self._board.squares_matrix[row_index][column_index]
            if square.is_free:
                free_squares.append(square)
            elif not square.is_free and self._current_piece.can_capture(square._current_piece):
                free_squares.append(square)
                break
            else:
                break

        for row_index in range(self._row_index + 1, self._board.num_of_rows, 1):
            square = self._board.squares_matrix[row_index][column_index]
            if square.is_free:
                free_squares.append(square)
            elif not square.is_free and self._current_piece.can_capture(square._current_piece):
                free_squares.append(square)
                break
            else:
                break

        return free_squares

    @property
    def real_row(self):
        return self._board.num_of_rows - self._row_index

    @property
    def column_letter(self):
        return chr(ord('a') + self._column_index)

    def __str__(self):
        return f"{self.column_letter}{self.real_row}"


class MoveTypes(Enum):
    MATE = 0
    CAPTURE = 1
    CHECK = 2
    REGULAR = 3  # TODO: support other types, maybe on passunt


class Move:
    def __init__(self, board: "Board", chosen_piece: "Optional[Piece]" = None,
                 chosen_square: "Optional[Square]" = None, move_turn: PIECE_COLOR = PIECE_COLOR.WHITE):
        self._board = board
        self._chosen_piece = chosen_piece
        self._destination_square = chosen_square
        self._types = None
        self._move_turn = move_turn

    @property
    def board(self):
        return self._board

    @board.setter
    def board(self, board):
        self._board = board

    @property
    def chosen_piece(self):
        return self._chosen_piece

    @chosen_piece.setter
    def chosen_piece(self, chosen_piece: "Piece"):
        self._chosen_piece = chosen_piece

    @property
    def destination_square(self):
        return self._destination_square

    @destination_square.setter
    def destination_square(self, destination_square: Square):  # TODO: check whether it's a valid square ?
        self._destination_square = destination_square

    def determine_types(self):
        self._types = []
        if self.is_valid():
            if not self._destination_square.is_free:  # Either a capture, or also a check and a mate
                self._types.append(MoveTypes.CAPTURE)
            else:
                self._types.append(MoveTypes.REGULAR)
            # TODO: here check whether it's also a check or a mate or so on

    def is_valid(self) -> bool:  # TODO: check somewhere whether the move is legal
        if self._destination_square is None or self._chosen_piece is None:
            return False  # The move hasn't been created yet
        return self._destination_square in self._chosen_piece.possible_squares()

    def execute(self):
        if self.is_valid():
            self._chosen_piece.move(self._destination_square)

    def __str__(self):
        if MoveTypes.REGULAR in self._types:
            return f"{self._chosen_piece.__str__()}{self._destination_square.__str__()}"
        else:
            return f"{self._chosen_piece.__str__()}X{self._destination_square.__str__()}"


class GameHandler:
    def __init__(self, board: "Board"):
        print("Creating a new game handler")
        self._board = board
        self._current_turn = PIECE_COLOR.WHITE
        self._current_move = Move(board)
        self._moves: List[Move] = []

    @property
    def board(self):
        return self._board

    @property
    def current_turn(self):
        return self._current_turn

    @property
    def current_move(self):
        return self._current_move

    @property
    def last_turn(self):
        try:
            return self._moves[-1]
        except IndexError:
            return None

    def set_chosen_piece(self, chosen_piece: "Piece"):
        if chosen_piece.color == self._current_turn:
            self._current_move.chosen_piece = chosen_piece

    def set_destination_square(self, destination_square: Square):
        if destination_square in self._current_move.chosen_piece.possible_squares():
            self._current_move.destination_square = destination_square
        else:
            print("illegal move!")
            self._current_move.chosen_piece = None
            self._current_move.destination_square = None

    def execute_move(self):
        if self._current_move.is_valid():
            print("executing move ...")
            self._current_move.execute()
            self.shift_turns()
            self._moves.append(self._current_move)
            self._current_move = Move(board=self._board)
        else:
            print("invalid move ...")

    def shift_turns(self):
        self._current_turn = PIECE_COLOR.BLACK if self._current_turn == PIECE_COLOR.WHITE else PIECE_COLOR.WHITE


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
            raise ValueError("The main board exceeds the screen's size")
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

    def static_evaluation(self) -> float:
        """
        Gives a static evaluation of which side has the advantage, by counting the points of the pieces
        of each side
        """
        white_sum, black_sum = 0, 0
        for row in self._squares:
            for square in row:
                if not square.is_free:
                    if square.current_piece.color == PIECE_COLOR.WHITE:
                        white_sum += square.current_piece.value
                    else:
                        black_sum += square.current_piece.value
        return white_sum - black_sum


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
                 value: float, location: Tuple[int, int], image: Union[Surface, pygame.Surface]):
        self._board = board
        self._color = piece_color
        self._value = value
        self._row, self._column = location
        self._square: Square = board.squares_matrix[self._row][self._column]
        self._image = image
        self._square.occupy(self)
        self._possible_squares = None

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

    @property
    def value(self):
        return self._value

    def draw(self):
        x = self._square.x + (self._square.width - self._image.get_rect().width) // 2
        y = self._square.y + (self._square.height - self._image.get_rect().height) // 2

        self._board.screen.blit(self._image, (x, y))
        pygame.display.update()

    def possible_squares(self):
        pass

    def move(self, destination_square: Square):
        self._square.free()
        self._row, self._column = destination_square.row_index, destination_square.column_index
        self._square = destination_square
        destination_square.occupy(self)
        self.draw()
        pygame.display.update()

    def can_capture(self, other_piece: "Piece") -> bool:
        return not isinstance(other_piece, King) and self._color != other_piece._color

    def __str__(self):
        return "Undefined piece behavior !"


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

    @property
    def has_moved(self):
        return self._has_moved

    def possible_squares(self):  # TODO: here
        possible_squares = []
        # CHECKING SIDES
        if self._square.has_right():
            if self._square.right_free() or self.can_capture(self._square.right_square.current_piece):
                possible_squares.append(self._square.right_square)

        if self._square.has_left():
            print("checking for left")
            if self._square.left_free() or self.can_capture(self._square.left_square.current_piece):
                print("checking inside the left")
                possible_squares.append(self._square.left_square)

        if self._square.has_up():
            if self._square.up_free() or self.can_capture(self._square.up_square.current_piece):
                possible_squares.append(self._square.up_square)

        if self._square.has_down():
            if self._square.down_free() or self.can_capture(self._square.down_square.current_piece):
                possible_squares.append(self._square.down_square)

        # CHECKING DIAGONALS
        if self._square.has_diagonal_up_left():
            if self._square.diagonal_up_left_free() or self.can_capture(self._square.diagonal_up_left.current_piece):
                possible_squares.append(self._square.diagonal_up_left)

        if self._square.has_diagonal_up_right():
            if self._square.diagonal_up_right_free() or self.can_capture(self._square.diagonal_up_right.current_piece):
                possible_squares.append(self._square.diagonal_up_right)

        if self._square.has_diagonal_down_left():
            if self._square.diagonal_down_left_free() or self.can_capture(
                    self._square.diagonal_down_left.current_piece):
                possible_squares.append(self._square.diagonal_down_left)

        if self._square.has_diagonal_down_right():
            if self._square.diagonal_down_right_free() or self.can_capture(
                    self._square.diagonal_down_right.current_piece):
                possible_squares.append(self._square.diagonal_down_right)
        self._possible_squares = possible_squares  # TODO: be careful of the memory sharing !

        return possible_squares

    def move(self, destination_square: Square):
        super(King, self).move(destination_square)
        self._has_moved = True

    def can_short_castle(self):  # TODO: CHECK WITH THE ROOKS, AND IF THE SQUARES IN BETWEEN ARE EMPTY
        if self._has_moved:
            return False

        if self._color == PIECE_COLOR.WHITE:
            rook_square: Square = self._board.squares_matrix[self._board.num_of_rows - 1][
                self._board.num_of_columns - 1]
            knight_square: Square = self._board.squares_matrix[self._board.num_of_rows - 1][
                self._board.num_of_columns - 2]
            bishop_square: Square = self._board.squares_matrix[self._board.num_of_rows - 1][
                self._board.num_of_columns - 3]
        else:
            rook_square: Square = self._board.squares_matrix[0][
                self._board.num_of_columns - 1]
            knight_square: Square = self._board.squares_matrix[0][
                self._board.num_of_columns - 2]
            bishop_square: Square = self._board.squares_matrix[0][
                self._board.num_of_columns - 3]

        if not knight_square.is_free or not bishop_square.is_free:  # The road to castle must be free !
            return False

        if isinstance(rook_square.current_piece, Rook):
            if not rook_square.current_piece.has_moved:
                return True
        return False

    def can_long_castle(self):  # TODO: implement it similarly to the short castle
        pass

    def can_capture(self, other_piece: Piece) -> bool:  # ADD WHETHER THE PIECE IS NOT PROTECTED
        return super(King, self).can_capture(other_piece)

    def __str__(self):
        return "K"


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
        possible_squares = self._square.free_squares_in_row(self._row) + self._square.free_squares_in_column(
            self._column) + self._square.full_free_diagonals()
        self._possible_squares = possible_squares  # TODO: be careful of the memory sharing !
        return possible_squares

    def __str__(self):
        return "Q"


class Rook(Piece):
    def __init__(self, board: Board, location: Tuple[int, int], piece_color: PIECE_COLOR):
        super().__init__(
            board=board,
            location=location,
            piece_color=piece_color,
            value=5,
            image=BLACK_ROOK if piece_color.value else WHITE_ROOK
        )
        self._has_moved = False

    @property
    def has_moved(self):
        return self._has_moved

    def possible_squares(self):
        possible_squares = self._square.free_squares_in_row(self._row) + self._square.free_squares_in_column(
            self._column)
        self._possible_squares = possible_squares  # TODO: be careful of the memory sharing !
        return possible_squares

    def move(self, destination_square: Square):
        super(Rook, self).move(destination_square)
        self._has_moved = True

    def __str__(self):
        return "R"


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
        possible_squares = self._square.full_free_diagonals()
        self._possible_squares = possible_squares  # TODO: be careful of the memory sharing !
        return possible_squares

    def __str__(self):
        return "B"


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
            if self._board.valid_row_index(row) and self._board.valid_column_index(column):
                square = self._board.squares_matrix[row][column]
                if square.is_free or super(Knight, self).can_capture(square.current_piece):
                    possible_positions.append(square)
        self._possible_squares = possible_positions  # TODO: be careful of the memory sharing !
        return possible_positions

    def __str__(self):
        return "N"


class Pawn(Piece):
    def __init__(self, board: Board, location: Tuple[int, int], piece_color: PIECE_COLOR):
        super().__init__(
            board=board,
            location=location,
            piece_color=piece_color,
            value=1,
            image=BLACK_PAWN if piece_color.value else WHITE_PAWN
        )

    def possible_captures(self):
        """ Returns a list of squares that the current pawn can capture"""
        possible_captures = []
        if self._color == PIECE_COLOR.WHITE:  # POSSIBLE CAPTURES FOR WHITE
            if self._square.has_diagonal_up_left() and not self._square.diagonal_up_left.is_free and self.can_capture(
                    self._square.diagonal_up_left.current_piece):
                possible_captures.append(self._square.diagonal_up_left)

            if self._square.has_diagonal_up_right() and not self._square.diagonal_up_right.is_free and self.can_capture(
                    self._square.diagonal_up_right.current_piece):
                possible_captures.append(self._square.diagonal_up_right)
        else:  # POSSIBLE CAPTURES FOR BLACK
            if self._square.has_diagonal_down_left() and not self._square.diagonal_down_left.is_free and self.can_capture(
                    self._square.diagonal_down_left.current_piece):
                possible_captures.append(self._square.diagonal_down_left)

            if self._square.has_diagonal_down_right() and not self._square.diagonal_down_right.is_free and self.can_capture(
                    self._square.diagonal_down_right.current_piece):
                possible_captures.append(self._square.diagonal_down_right)

        return possible_captures

    def possible_squares(self) -> List[Square]:
        """ Returns a list of all of the squares that the pawn can go or capture"""
        possible_squares: List[Square] = []
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

        self._possible_squares = possible_squares + self.possible_captures()  # TODO: be careful of the memory sharing !
        return self._possible_squares


main_page = Page(1100, 900, "Chess Engine V1.0")
main_board = Board(main_page.window, pygame.Rect(165, 60, 780, 780), colors=(WHITE, BROWN))
pieces = generate_pieces(main_board)
main_board.draw()

game_handler = GameHandler(
    board=main_board
)


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
    global MOVING_PIECE
    while RUNNING:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                RUNNING = False
            elif event.type == pygame.MOUSEBUTTONDOWN:
                print(f"Eval Bar: {main_board.static_evaluation()}")
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
