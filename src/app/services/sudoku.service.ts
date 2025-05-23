import { Injectable } from '@angular/core';
import { HttpClient } from '@angular/common/http';
import { Observable } from 'rxjs';
import { map } from 'rxjs/operators';

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
}

@Injectable({
  providedIn: 'root'
})
export class SudokuService {
  private apiUrl = 'https://sudoku-api.vercel.app/api/dosuku';

  constructor(private http: HttpClient) { }

  // Get a new Sudoku board with specified difficulty
  getNewBoard(difficulty?: string): Observable<SudokuBoard> {
    let query = '{newboard(limit:1){grids{value,solution,difficulty},results,message}}';
    
    // If a difficulty is specified, we'll filter for that difficulty
    // Note: The API doesn't support difficulty filtering directly, so we'll need to handle that client-side
    
    return this.http.get<SudokuResponse>(`${this.apiUrl}?query=${query}`)
      .pipe(
        map(response => {
          const grid = response.newboard.grids[0];
          
          // If a specific difficulty was requested but doesn't match, we'll handle that later
          // by requesting a new board until we get the right difficulty
          
          return {
            grid: grid.value,
            solution: grid.solution,
            difficulty: grid.difficulty
          };
        })
      );
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
