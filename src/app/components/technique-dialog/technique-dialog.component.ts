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
        <div class="button-container">
          <!-- Primary action buttons -->
          <div class="primary-actions">
            <button mat-raised-button color="primary" [mat-dialog-close]="'apply'" class="apply-button">
              Apply
            </button>
            <button mat-button mat-dialog-close class="cancel-button">Cancel</button>
          </div>

          <!-- Skip options -->
          <div class="skip-options" *ngIf="data.showSkipOptions">
            <div class="skip-buttons">
              <button mat-stroked-button [mat-dialog-close]="'skip-this'" class="skip-button">
                Skip This
              </button>
              <button mat-stroked-button [mat-dialog-close]="'skip-type'" class="skip-button">
                Skip All {{ data.technique }}
              </button>
              <button mat-stroked-button [mat-dialog-close]="'skip-all'" class="skip-button">
                Skip to End
              </button>
            </div>
          </div>
        </div>
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
      padding: 0 5px;
    }

    :host-context(.dark-theme) .technique-dialog-container {
      color: #f5f5f5;
    }

    mat-dialog-content {
      margin-bottom: 1.5rem;
      padding: 0;
    }

    .technique-details {
      margin: 1rem 0;
      padding: 0.75rem;
      background: rgba(0, 0, 0, 0.04);
      border-radius: 8px;
    }

    .technique-details p {
      margin: 0.5rem 0;
    }

    .teaching-progress {
      margin-top: 1rem;
      text-align: center;
      font-style: italic;
      color: rgba(0, 0, 0, 0.6);
    }

    /* Button container layout */
    .button-container {
      display: flex;
      flex-direction: column;
      gap: 1rem;
      width: 100%;
    }

    /* Primary actions styling */
    .primary-actions {
      display: flex;
      gap: 0.75rem;
      justify-content: flex-end;
    }

    .apply-button {
      min-width: 100px;
    }

    /* Skip options styling */
    .skip-options {
      border-top: 1px solid rgba(0, 0, 0, 0.12);
      padding-top: 1rem;
    }

    .skip-buttons {
      display: flex;
      flex-direction: column;
      gap: 0.5rem;
    }

    .skip-button {
      width: 100%;
      justify-content: center;
    }

    /* Dark theme support */
    :host-context(.dark-theme) {
      .technique-details {
        background: rgba(255, 255, 255, 0.1);
      }

      .teaching-progress {
        color: rgba(255, 255, 255, 0.7);
      }

      .skip-options {
        border-top-color: rgba(255, 255, 255, 0.12);
      }
    }

    /* Mobile responsive design */
    @media (max-width: 600px) {
      .technique-dialog-container {
        padding: 1rem 0.75rem;
      }

      mat-dialog-content {
        margin-bottom: 1rem;
      }

      .button-container {
        gap: 0.75rem;
      }

      .primary-actions {
        flex-direction: column-reverse;
        gap: 0.5rem;
      }

      .apply-button,
      .cancel-button {
        width: 100%;
        margin: 0;
      }

      .skip-buttons {
        gap: 0.5rem;
      }

      .skip-button {
        padding: 0.5rem;
      }
    }
  `]
})
export class TechniqueDialogComponent {
  constructor(
    public dialogRef: MatDialogRef<TechniqueDialogComponent>,
    @Inject(MAT_DIALOG_DATA) public data: TechniqueDialogData
  ) {}
}