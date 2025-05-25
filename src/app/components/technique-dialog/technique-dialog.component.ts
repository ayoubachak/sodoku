import { Component, Inject } from '@angular/core';
import { CommonModule } from '@angular/common';
import { MAT_DIALOG_DATA, MatDialogRef, MatDialogModule } from '@angular/material/dialog';
import { MatButtonModule } from '@angular/material/button';
import { SudokuTechnique } from '../../services/technique-coach.service';

export interface TechniqueDialogData {
  title: string;
  message: string;
  technique: SudokuTechnique;
  value?: number;
  position?: { row: number, col: number };
  showRevealOption?: boolean;
  // Add properties for showing skip buttons
  showSkipOptions?: boolean;
  isTeachingMode?: boolean;
  techniqueCount?: number;
  currentIndex?: number;
}

@Component({
  selector: 'app-technique-dialog',
  standalone: true,
  imports: [CommonModule, MatDialogModule, MatButtonModule],
  template: `
    <div class="technique-dialog-container">
      <h2 mat-dialog-title>{{ data.title }}</h2>
      <mat-dialog-content>
        <p [innerHTML]="data.message"></p>
        <div class="technique-details" *ngIf="data.position">
          <p class="position">
            Position: Row {{ data.position.row + 1 }}, Column {{ data.position.col + 1 }}
          </p>
          <p class="value" *ngIf="data.value">
            Value: {{ data.value }}
          </p>
        </div>
        <div class="teaching-progress" *ngIf="data.isTeachingMode && data.techniqueCount && data.currentIndex !== undefined">
          <p class="progress-text">Technique {{ data.currentIndex + 1 }} of {{ data.techniqueCount }}</p>
        </div>
      </mat-dialog-content>
      <mat-dialog-actions>
        <button mat-button mat-dialog-close>Cancel</button>
        <button mat-button *ngIf="data.showRevealOption" [mat-dialog-close]="'reveal'">Reveal Value</button>
        
        <!-- Skip buttons for teaching mode -->
        <div class="skip-actions" *ngIf="data.showSkipOptions">
          <button mat-button [mat-dialog-close]="'skip-this'" class="skip-button">
            Skip
          </button>
          <button mat-button [mat-dialog-close]="'skip-apply'" class="skip-button apply-skip-button">
            Skip & Apply
          </button>
          <button mat-button [mat-dialog-close]="'skip-type'" class="skip-button">
            Skip All {{ data.technique }}
          </button>
          <button mat-button [mat-dialog-close]="'skip-all'" class="skip-button">
            Skip to End
          </button>
        </div>
        
        <button mat-raised-button color="primary" [mat-dialog-close]="'apply'">Apply</button>
      </mat-dialog-actions>
    </div>
  `,
  styles: [`
    :host {
      display: block;
    }

    ::ng-deep .technique-dialog .mat-mdc-dialog-surface {
      background-color: rgba(255, 255, 255, 0.95) !important;
      backdrop-filter: blur(5px);
      position: absolute;
      top: 10px;
      right: 10px;
      max-width: 400px;
      max-height: calc(100vh - 20px);
      width: 95%;
      overflow: auto;
      border-radius: 12px;
      box-shadow: 0 8px 32px rgba(0, 0, 0, 0.2);
      border: 1px solid rgba(255, 255, 255, 0.3);
    }

    ::ng-deep .dark-theme .technique-dialog .mat-mdc-dialog-surface {
      background-color: rgba(45, 45, 45, 0.95) !important;
      border: 1px solid rgba(255, 255, 255, 0.1);
    }

    .technique-dialog-container {
      color: #333;
    }

    :host-context(.dark-theme) .technique-dialog-container {
      color: #f5f5f5;
    }

    :host-context(.dark-theme) h2 {
      color: #f3f4f6;
    }

    :host-context(.dark-theme) mat-dialog-content {
      color: #e5e7eb;
    }

    .technique-details {
      margin: 16px 0;
      padding: 8px 16px;
      background-color: #f5f5f5;
      border-left: 4px solid #3f51b5;
      border-radius: 4px;
    }
    
    :host-context(.dark-theme) .technique-details {
      background-color: #374151;
      border-left-color: #7986cb;
      color: #e5e7eb;
    }
    
    .position, .value {
      margin: 6px 0;
    }
    
    mat-dialog-actions {
      display: flex;
      justify-content: flex-end;
      flex-wrap: wrap;
      gap: 8px;
    }
    
    .skip-actions {
      display: flex;
      gap: 8px;
      margin-right: auto;
      flex-wrap: wrap;
    }
    
    .skip-button {
      font-size: 0.85rem;
      padding: 0 8px;
    }
    
    .apply-skip-button {
      background-color: #e3f2fd;
      border: 1px solid #bbdefb;
    }
    
    :host-context(.dark-theme) .apply-skip-button {
      background-color: #0d47a1;
      color: white;
    }

    :host-context(.dark-theme) .skip-button {
      color: #e5e7eb;
    }

    :host-context(.dark-theme) button[mat-button]:not(.apply-skip-button) {
      color: #e5e7eb;
    }
    
    :host-context(.dark-theme) button[mat-button]:hover {
      background-color: rgba(255, 255, 255, 0.08);
    }
    
    .teaching-progress {
      margin-top: 16px;
      text-align: center;
      color: #666;
      font-style: italic;
    }
    
    :host-context(.dark-theme) .teaching-progress {
      color: #aaa;
    }
    
    .progress-text {
      margin: 0;
      font-size: 0.9rem;
    }
  `]
})
export class TechniqueDialogComponent {
  constructor(
    public dialogRef: MatDialogRef<TechniqueDialogComponent>,
    @Inject(MAT_DIALOG_DATA) public data: TechniqueDialogData
  ) {}
}