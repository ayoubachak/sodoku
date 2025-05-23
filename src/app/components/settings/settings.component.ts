import { Component } from '@angular/core';
import { CommonModule } from '@angular/common';
import { FormsModule } from '@angular/forms';
import { MatButtonModule } from '@angular/material/button';
import { MatCardModule } from '@angular/material/card';
import { MatSlideToggleModule } from '@angular/material/slide-toggle';
import { MatSelectModule } from '@angular/material/select';
import { Router } from '@angular/router';

@Component({
  selector: 'app-settings',
  standalone: true,
  imports: [
    CommonModule, 
    FormsModule, 
    MatButtonModule, 
    MatCardModule, 
    MatSlideToggleModule,
    MatSelectModule
  ],
  templateUrl: './settings.component.html',
  styleUrl: './settings.component.css'
})
export class SettingsComponent {
  showMistakes: boolean = true;
  showTimer: boolean = true;
  highlightSameNumbers: boolean = true;
  theme: string = 'light';

  constructor(private router: Router) {
    // Load settings from localStorage if available
    const savedSettings = localStorage.getItem('sudokuSettings');
    if (savedSettings) {
      const settings = JSON.parse(savedSettings);
      this.showMistakes = settings.showMistakes ?? true;
      this.showTimer = settings.showTimer ?? true;
      this.highlightSameNumbers = settings.highlightSameNumbers ?? true;
      this.theme = settings.theme ?? 'light';
    }
  }

  saveSettings(): void {
    const settings = {
      showMistakes: this.showMistakes,
      showTimer: this.showTimer,
      highlightSameNumbers: this.highlightSameNumbers,
      theme: this.theme
    };
    
    localStorage.setItem('sudokuSettings', JSON.stringify(settings));
    this.router.navigate(['/menu']);
  }

  cancel(): void {
    this.router.navigate(['/menu']);
  }
}
