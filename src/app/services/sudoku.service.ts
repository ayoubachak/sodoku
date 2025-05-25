import { Injectable } from '@angular/core';
import { HttpClient } from '@angular/common/http';
import { Observable, of } from 'rxjs';
import { map, catchError } from 'rxjs/operators';

export interface SudokuResponse {
  newboard: {
    grids: {
      value: number[][];
      solution: number[][];
      difficulty: string;
    }[];
    results: number;
    message: string;
  };
}

export interface SudokuBoard {
  grid: number[][];
  solution: number[][];
  difficulty: string;
  notes?: number[][][]; // 9x9 grid of arrays containing possible numbers
}

export interface CellData {
  value: number;
  notes: number[];
  isOriginal: boolean;
}

@Injectable({
  providedIn: 'root'
})
export class SudokuService {
  private apiUrl = 'https://sudoku-api.vercel.app/api/dosuku';

  constructor(private http: HttpClient) { }

  // Get a new Sudoku board with specified difficulty
  getNewBoard(difficulty?: string): Observable<SudokuBoard> {
    // GraphQL query to get a complete board with value, solution, and difficulty
    const query = '{newboard(limit:1){grids{value,solution,difficulty},results,message}}';
    
    return this.http.get<SudokuResponse>(`${this.apiUrl}?query=${encodeURIComponent(query)}`)
      .pipe(
        map(response => {
          const grid = response.newboard.grids[0];
          
          return {
            grid: grid.value,
            solution: grid.solution,
            difficulty: grid.difficulty
          };
        }),
        catchError(error => {
          console.error('Error fetching Sudoku board:', error);
          // Return a fallback board if API fails
          return of(this.getFallbackBoard());
        })
      );
  }

  // Fallback board in case API fails
  private getFallbackBoard(): SudokuBoard {
    return {
      grid: [
        [5, 3, 0, 0, 7, 0, 0, 0, 0],
        [6, 0, 0, 1, 9, 5, 0, 0, 0],
        [0, 9, 8, 0, 0, 0, 0, 6, 0],
        [8, 0, 0, 0, 6, 0, 0, 0, 3],
        [4, 0, 0, 8, 0, 3, 0, 0, 1],
        [7, 0, 0, 0, 2, 0, 0, 0, 6],
        [0, 6, 0, 0, 0, 0, 2, 8, 0],
        [0, 0, 0, 4, 1, 9, 0, 0, 5],
        [0, 0, 0, 0, 8, 0, 0, 7, 9]
      ],
      solution: [
        [5, 3, 4, 6, 7, 8, 9, 1, 2],
        [6, 7, 2, 1, 9, 5, 3, 4, 8],
        [1, 9, 8, 3, 4, 2, 5, 6, 7],
        [8, 5, 9, 7, 6, 1, 4, 2, 3],
        [4, 2, 6, 8, 5, 3, 7, 9, 1],
        [7, 1, 3, 9, 2, 4, 8, 5, 6],
        [9, 6, 1, 5, 3, 7, 2, 8, 4],
        [2, 8, 7, 4, 1, 9, 6, 3, 5],
        [3, 4, 5, 2, 8, 6, 1, 7, 9]
      ],
      difficulty: 'Medium'
    };
  }

  // Helper method to check if a move is valid
  isValidMove(grid: number[][], row: number, col: number, num: number): boolean {
    // Check row
    for (let x = 0; x < 9; x++) {
      if (grid[row][x] === num) {
        return false;
      }
    }

    // Check column
    for (let y = 0; y < 9; y++) {
      if (grid[y][col] === num) {
        return false;
      }
    }

    // Check 3x3 box
    const startRow = Math.floor(row / 3) * 3;
    const startCol = Math.floor(col / 3) * 3;
    for (let i = 0; i < 3; i++) {
      for (let j = 0; j < 3; j++) {
        if (grid[startRow + i][startCol + j] === num) {
          return false;
        }
      }
    }

    return true;
  }

  // Method to check if the board is solved
  isSolved(grid: number[][]): boolean {
    for (let i = 0; i < 9; i++) {
      for (let j = 0; j < 9; j++) {
        if (grid[i][j] === 0) {
          return false;
        }
      }
    }
    return true;
  }
}
