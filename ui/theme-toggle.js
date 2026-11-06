document.addEventListener('DOMContentLoaded', () => {
  const themeToggle = document.getElementById('theme-toggle');
  const mobileThemeToggle = document.getElementById('mobile-theme-toggle'); // For home page mobile menu

  // Function to set the theme
  const setTheme = (theme) => {
    document.documentElement.setAttribute('data-theme', theme);
    try { localStorage.setItem('theme', theme); } catch (e) { /* storage blocked */ }
    updateToggleIcon(theme);
  };

  // Function to update the toggle icon based on the current theme
  const updateToggleIcon = (theme) => {
    const icon = theme === 'dark' ? 'light_mode' : 'dark_mode';
    if (themeToggle) {
      themeToggle.querySelector('.material-symbols-outlined').textContent = icon;
    }
    if (mobileThemeToggle) {
      mobileThemeToggle.querySelector('.material-symbols-outlined').textContent = icon;
    }
  };

  // Initialize theme from localStorage or default to light
  let savedTheme = null;
  try { savedTheme = localStorage.getItem('theme'); } catch (e) { /* storage blocked */ }
  if (!savedTheme) savedTheme = window.matchMedia && matchMedia('(prefers-color-scheme: dark)').matches ? 'dark' : 'light';   // follow the system until the visitor chooses
  setTheme(savedTheme);

  // Event listener for desktop theme toggle
  if (themeToggle) {
    themeToggle.addEventListener('click', () => {
      const currentTheme = document.documentElement.getAttribute('data-theme');
      const newTheme = currentTheme === 'light' ? 'dark' : 'light';
      setTheme(newTheme);
    });
  }

  // Event listener for mobile theme toggle (home page specific)
  if (mobileThemeToggle) {
    mobileThemeToggle.addEventListener('click', () => {
      const currentTheme = document.documentElement.getAttribute('data-theme');
      const newTheme = currentTheme === 'light' ? 'dark' : 'light';
      setTheme(newTheme);
    });
  }
});
