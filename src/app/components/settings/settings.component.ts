import { Component, OnInit } from '@angular/core';
import { CommonModule } from '@angular/common';
import { FormsModule } from '@angular/forms';
import { MatButtonModule } from '@angular/material/button';
import { MatCardModule } from '@angular/material/card';
import { MatSlideToggleModule } from '@angular/material/slide-toggle';
import { MatSelectModule } from '@angular/material/select';
import { MatIconModule } from '@angular/material/icon';
import { Router } from '@angular/router';
import { ThemeService } from '../../services/theme.service';

@Component({
  selector: 'app-settings',
  standalone: true,
  imports: [
    CommonModule, 
    FormsModule, 
    MatButtonModule, 
    MatCardModule, 
    MatSlideToggleModule,
    MatSelectModule,
    MatIconModule
  ],
  templateUrl: './settings.component.html',
  styleUrl: './settings.component.css'
})
export class SettingsComponent implements OnInit {
  showMistakes: boolean = true;
  showTimer: boolean = true;
  highlightSameNumbers: boolean = true;
  theme: string = 'light';
  useApi: boolean = true;
  localDifficulty: string = 'medium';

  constructor(private router: Router, private themeService: ThemeService) {}

  ngOnInit(): void {
    // Subscribe to theme changes
    this.themeService.isDarkMode$.subscribe(isDark => {
      this.theme = isDark ? 'dark' : 'light';
    });

    // Load settings from localStorage if available
    const savedSettings = localStorage.getItem('sudokuSettings');
    if (savedSettings) {
      const settings = JSON.parse(savedSettings);
      this.showMistakes = settings.showMistakes ?? true;
      this.showTimer = settings.showTimer ?? true;
      this.highlightSameNumbers = settings.highlightSameNumbers ?? true;
      this.useApi = settings.useApi ?? true;
      this.localDifficulty = settings.localDifficulty ?? 'medium';
    }
  }

  onThemeChange(): void {
    this.themeService.setTheme(this.theme === 'dark');
  }

  onApiToggle(): void {
    // Save the setting immediately when toggled
    this.saveSettings();
  }

  saveSettings(): void {
    const settings = {
      showMistakes: this.showMistakes,
      showTimer: this.showTimer,
      highlightSameNumbers: this.highlightSameNumbers,
      theme: this.theme,
      useApi: this.useApi,
      localDifficulty: this.localDifficulty
    };
    
    localStorage.setItem('sudokuSettings', JSON.stringify(settings));
    
    this.router.navigate(['/menu']);
  }

  cancel(): void {
    this.router.navigate(['/menu']);
  }
}
