import { Injectable } from '@angular/core';
import * as tf from '@tensorflow/tfjs';
import { BehaviorSubject, Observable } from 'rxjs';

// Define the available Sudoku solving techniques
export enum SudokuTechnique {
  NAKED_SINGLE = 'Naked Single',
  HIDDEN_SINGLE = 'Hidden Single',
  NAKED_PAIR = 'Naked Pair',
  HIDDEN_PAIR = 'Hidden Pair',
  NAKED_TRIPLE = 'Naked Triple',
  HIDDEN_TRIPLE = 'Hidden Triple',
  X_WING = 'X-Wing',
  SWORDFISH = 'Swordfish',
  Y_WING = 'Y-Wing'
}

// Interface for technique detection result
export interface TechniqueDetection {
  technique: SudokuTechnique;
  targetCell?: { row: number, col: number };
  relatedCells?: { row: number, col: number }[];
  explanation: string;
  value?: number; // The digit to be placed
  confidence: number; // Model confidence score
  detailExplanation?: string; // Additional detailed explanation for teaching mode
  prerequisites?: string[]; // Any prerequisites needed (like "notes required")
}

// Interface for technique usage statistics
export interface TechniqueStats {
  technique: SudokuTechnique;
  usageCount: number;
  successRate: number; // How often the user applies the technique correctly
}

// Interface for a user's technique profile
export interface UserTechniqueProfile {
  strengths: SudokuTechnique[];
  weaknesses: SudokuTechnique[];
  lastUpdated: Date;
}

@Injectable({
  providedIn: 'root'
})
export class TechniqueCoachService {
  private model: tf.LayersModel | null = null;
  private modelLoaded = false;
  private modelLoading = false;
  private readonly MODEL_URL = 'assets/models/sudoku_coach/model.json';
  
  private userProfileSubject = new BehaviorSubject<UserTechniqueProfile>({
    strengths: [],
    weaknesses: [],
    lastUpdated: new Date()
  });
  
  // Track technique usage statistics
  private techniqueStats: Map<SudokuTechnique, TechniqueStats> = new Map();

  constructor() {
    this.initializeStats();
    this.loadModel();
  }

  private initializeStats(): void {
    // Initialize statistics for each technique
    Object.values(SudokuTechnique).forEach(technique => {
      this.techniqueStats.set(technique as SudokuTechnique, {
        technique: technique as SudokuTechnique,
        usageCount: 0,
        successRate: 0
      });
    });
    
    // Load saved stats from localStorage if available
    const savedStats = localStorage.getItem('sudoku-technique-stats');
    if (savedStats) {
      try {
        const parsedStats = JSON.parse(savedStats);
        Object.entries(parsedStats).forEach(([technique, stats]) => {
          this.techniqueStats.set(technique as SudokuTechnique, stats as TechniqueStats);
        });
      } catch (error) {
        console.error('Failed to parse saved technique statistics:', error);
      }
    }
  }

  private async loadModel(): Promise<void> {
    if (this.modelLoaded || this.modelLoading) {
      return;
    }

    this.modelLoading = true;
    
    try {
      // For this implementation, we'll simulate the model loading
      // In a real implementation, you would load the TF.js model like this:
      // this.model = await tf.loadLayersModel(this.MODEL_URL);
      
      // Simulate model loading with a timeout
      await new Promise(resolve => setTimeout(resolve, 1000));
      
      this.modelLoaded = true;
      console.log('Technique detection model loaded successfully');
    } catch (error) {
      console.error('Failed to load technique detection model:', error);
    } finally {
      this.modelLoading = false;
    }
  }

  /**
   * Analyzes the current board state to detect applicable techniques
   * @param board The current Sudoku board (9x9 grid)
   * @param notes The notes/candidates for each cell
   * @returns An array of detected techniques with explanations
   */
  async detectTechniques(board: number[][], notes: number[][][]): Promise<TechniqueDetection[]> {
    await this.ensureModelLoaded();
    
    // In a real implementation, you would:
    // 1. Convert the board to the appropriate tensor format
    // 2. Run model.predict() to get technique predictions
    // 3. Process the predictions into meaningful insights
    
    // For now, we'll implement a rule-based detection system
    return this.detectTechniquesRuleBased(board, notes);
  }
  
  /**
   * Rule-based technique detection (fallback when model isn't available)
   */
  private detectTechniquesRuleBased(board: number[][], notes: number[][][]): TechniqueDetection[] {
    const detections: TechniqueDetection[] = [];
    const hasEmptyCellsWithoutNotes = this.checkEmptyCellsWithoutNotes(board, notes);
    
    // Check for Naked Singles (cells with only one candidate)
    for (let row = 0; row < 9; row++) {
      for (let col = 0; col < 9; col++) {
        // Skip filled cells
        if (board[row][col] !== 0) continue;
        
        // Check if this is a naked single (only one candidate)
        if (notes[row][col].length === 1) {
          const value = notes[row][col][0];
          detections.push({
            technique: SudokuTechnique.NAKED_SINGLE,
            targetCell: { row, col },
            explanation: `Only the number ${value} can go in this cell, as all other candidates have been eliminated.`,
            value,
            confidence: 0.95,
            detailExplanation: `In R${row+1}C${col+1}, all other candidates have been eliminated, leaving ${value} as the only possible value. This is called a "Naked Single" because the single candidate is directly visible.`
          });
        }
      }
    }
    
    // Check for Hidden Singles (a number only appears once as a candidate in a row/column/box)
    // Row-based hidden singles
    for (let row = 0; row < 9; row++) {
      const digitAppearances = new Map<number, { count: number, col: number }>();
      
      // Count appearances of each digit as a candidate in this row
      for (let col = 0; col < 9; col++) {
        if (board[row][col] !== 0) continue;
        
        for (const digit of notes[row][col]) {
          if (!digitAppearances.has(digit)) {
            digitAppearances.set(digit, { count: 1, col });
          } else {
            const current = digitAppearances.get(digit)!;
            digitAppearances.set(digit, { count: current.count + 1, col: current.col });
          }
        }
      }
      
      // Check if any digit appears only once
      for (const [digit, info] of digitAppearances.entries()) {
        if (info.count === 1) {
          detections.push({
            technique: SudokuTechnique.HIDDEN_SINGLE,
            targetCell: { row, col: info.col },
            explanation: `Only this cell in row ${row + 1} can contain the number ${digit}, as it's eliminated from all other cells.`,
            value: digit,
            confidence: 0.9,
            detailExplanation: `In row ${row+1}, the number ${digit} can only go in column ${info.col+1} because it's eliminated from all other cells in this row. This is called a "Hidden Single" because the unique placement isn't immediately obvious.`
          });
        }
      }
    }
    
    // Column-based hidden singles
    for (let col = 0; col < 9; col++) {
      const digitAppearances = new Map<number, { count: number, row: number }>();
      
      // Count appearances of each digit as a candidate in this column
      for (let row = 0; row < 9; row++) {
        if (board[row][col] !== 0) continue;
        
        for (const digit of notes[row][col]) {
          if (!digitAppearances.has(digit)) {
            digitAppearances.set(digit, { count: 1, row });
          } else {
            const current = digitAppearances.get(digit)!;
            digitAppearances.set(digit, { count: current.count + 1, row: current.row });
          }
        }
      }
      
      // Check if any digit appears only once
      for (const [digit, info] of digitAppearances.entries()) {
        if (info.count === 1) {
          detections.push({
            technique: SudokuTechnique.HIDDEN_SINGLE,
            targetCell: { row: info.row, col },
            explanation: `Only this cell in column ${col + 1} can contain the number ${digit}, as it's eliminated from all other cells.`,
            value: digit,
            confidence: 0.9,
            detailExplanation: `In column ${col+1}, the number ${digit} can only go in row ${info.row+1} because it's eliminated from all other cells in this column. This is a column-based "Hidden Single".`
          });
        }
      }
    }
    
    // Box-based hidden singles
    for (let boxRow = 0; boxRow < 3; boxRow++) {
      for (let boxCol = 0; boxCol < 3; boxCol++) {
        const boxStartRow = boxRow * 3;
        const boxStartCol = boxCol * 3;
        const digitAppearances = new Map<number, { count: number, row: number, col: number }>();
        
        // Count appearances of each digit as a candidate in this box
        for (let r = boxStartRow; r < boxStartRow + 3; r++) {
          for (let c = boxStartCol; c < boxStartCol + 3; c++) {
            if (board[r][c] !== 0) continue;
            
            for (const digit of notes[r][c]) {
              if (!digitAppearances.has(digit)) {
                digitAppearances.set(digit, { count: 1, row: r, col: c });
              } else {
                const current = digitAppearances.get(digit)!;
                digitAppearances.set(digit, { count: current.count + 1, row: current.row, col: current.col });
              }
            }
          }
        }
        
        // Check if any digit appears only once
        for (const [digit, info] of digitAppearances.entries()) {
          if (info.count === 1) {
            detections.push({
              technique: SudokuTechnique.HIDDEN_SINGLE,
              targetCell: { row: info.row, col: info.col },
              explanation: `Only this cell in box ${boxRow * 3 + boxCol + 1} can contain the number ${digit}.`,
              value: digit,
              confidence: 0.9,
              detailExplanation: `In the ${boxRow * 3 + boxCol + 1} box, the number ${digit} can only go at R${info.row+1}C${info.col+1} because it's eliminated from all other cells in this box. This is a box-based "Hidden Single".`
            });
          }
        }
      }
    }
    
    // Add prerequisites indicator if needed
    if (hasEmptyCellsWithoutNotes) {
      detections.forEach(detection => {
        detection.prerequisites = ['You need to fill in the notes (candidates) for empty cells to use this technique effectively.'];
      });
    }
    
    return detections;
  }
  
  /**
   * Check if there are empty cells without notes
   */
  private checkEmptyCellsWithoutNotes(board: number[][], notes: number[][][]): boolean {
    for (let row = 0; row < 9; row++) {
      for (let col = 0; col < 9; col++) {
        if (board[row][col] === 0 && notes[row][col].length === 0) {
          return true;
        }
      }
    }
    return false;
  }
  
  /**
   * Gets a hint for the player based on the current board state
   */
  async getHint(board: number[][], notes: number[][][]): Promise<TechniqueDetection | null> {
    const techniques = await this.detectTechniques(board, notes);
    
    if (techniques.length === 0) {
      return null;
    }
    
    // Sort by confidence and choose the highest confidence technique
    techniques.sort((a, b) => b.confidence - a.confidence);
    return techniques[0];
  }
  
  /**
   * Checks if a specific technique can be applied to a specific cell
   * @param board The current Sudoku board
   * @param notes The notes/candidates for each cell
   * @param row The row of the cell to check
   * @param col The column of the cell to check
   * @returns A technique that can be applied to this cell, or null if none found
   */
  async getHintForCell(board: number[][], notes: number[][][], row: number, col: number): Promise<TechniqueDetection | null> {
    // Skip if the cell is already filled
    if (board[row][col] !== 0) return null;
    
    const techniques = await this.detectTechniques(board, notes);
    
    // Check for techniques targeting this specific cell
    const cellTechniques = techniques.filter(t => 
      t.targetCell && t.targetCell.row === row && t.targetCell.col === col
    );
    
    if (cellTechniques.length > 0) {
      // Sort by confidence and choose the highest confidence technique
      cellTechniques.sort((a, b) => b.confidence - a.confidence);
      return cellTechniques[0];
    }
    
    // If no techniques found for this cell, return null
    return null;
  }

  /**
   * Records that a technique was used by the player
   */
  recordTechniqueUsage(technique: SudokuTechnique, successful: boolean): void {
    const stats = this.techniqueStats.get(technique);
    if (stats) {
      stats.usageCount++;
      const totalSuccess = stats.successRate * (stats.usageCount - 1) + (successful ? 1 : 0);
      stats.successRate = totalSuccess / stats.usageCount;
      
      // Save stats to localStorage
      this.saveTechniqueStats();
      
      // Update user profile
      this.updateUserProfile();
    }
  }
  
  /**
   * Saves technique usage statistics to localStorage
   */
  private saveTechniqueStats(): void {
    try {
      const statsObj = Object.fromEntries(this.techniqueStats);
      localStorage.setItem('sudoku-technique-stats', JSON.stringify(statsObj));
    } catch (error) {
      console.error('Failed to save technique statistics:', error);
    }
  }
  
  /**
   * Updates the user's technique profile based on statistics
   */
  private updateUserProfile(): void {
    const stats = Array.from(this.techniqueStats.values());
    
    // Sort by success rate and usage count
    const sortedStats = [...stats]
      .filter(s => s.usageCount > 0) // Only consider techniques that have been used
      .sort((a, b) => b.successRate - a.successRate);
    
    // Get top 3 strengths (highest success rates)
    const strengths = sortedStats.slice(0, 3).map(s => s.technique);
    
    // Get top 3 weaknesses (lowest success rates)
    const weaknesses = [...sortedStats]
      .sort((a, b) => a.successRate - b.successRate)
      .slice(0, 3)
      .map(s => s.technique);
    
    const profile = {
      strengths,
      weaknesses,
      lastUpdated: new Date()
    };
    
    this.userProfileSubject.next(profile);
    
    // Save to localStorage
    try {
      localStorage.setItem('sudoku-user-profile', JSON.stringify(profile));
    } catch (error) {
      console.error('Failed to save user technique profile:', error);
    }
  }
  
  /**
   * Gets the user's technique profile as an observable
   */
  getUserProfile(): Observable<UserTechniqueProfile> {
    return this.userProfileSubject.asObservable();
  }
  
  /**
   * Recommends a difficulty level based on the user's profile
   */
  recommendDifficulty(): 'easy' | 'medium' | 'hard' | 'expert' {
    const profile = this.userProfileSubject.value;
    const advancedTechniques = [
      SudokuTechnique.X_WING,
      SudokuTechnique.SWORDFISH,
      SudokuTechnique.Y_WING
    ];
    
    const hasAdvancedWeakness = profile.weaknesses.some(t => advancedTechniques.includes(t));
    const hasAdvancedStrength = profile.strengths.some(t => advancedTechniques.includes(t));
    
    if (hasAdvancedWeakness) {
      return 'hard'; // Practice difficult techniques
    } else if (hasAdvancedStrength) {
      return 'expert'; // Challenge with even harder puzzles
    } else if (profile.weaknesses.includes(SudokuTechnique.HIDDEN_PAIR)) {
      return 'medium'; // Practice intermediate techniques
    } else {
      return 'easy'; // Start with basics
    }
  }
  
  /**
   * Helper to ensure the model is loaded before using it
   */
  private async ensureModelLoaded(): Promise<void> {
    if (!this.modelLoaded && !this.modelLoading) {
      await this.loadModel();
    }
    
    // If model is still loading, wait for it
    while (this.modelLoading) {
      await new Promise(resolve => setTimeout(resolve, 100));
    }
  }
  
  /**
   * Gets all technique usage statistics
   */
  getTechniqueStats(): TechniqueStats[] {
    return Array.from(this.techniqueStats.values());
  }
}
