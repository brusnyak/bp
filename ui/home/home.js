document.addEventListener('DOMContentLoaded', () => {
  const mainNav = document.getElementById('main-nav');
  let lastScrollY = window.scrollY;
  const header = document.getElementById('main-nav'); // Assuming main-nav is your header

  // Header scroll behavior
  window.addEventListener('scroll', () => {
    const isInThesisMode = body.classList.contains('thesis-mode'); // Check if in thesis mode

    if (!isInThesisMode) { // Only apply header scroll behavior in default mode
      if (window.scrollY > lastScrollY && window.scrollY > 100) {
        // Scrolling down and past a certain threshold
        header.classList.add('nav-hidden');
      } else {
        // Scrolling up or at the top
        header.classList.remove('nav-hidden');
      }
    } else {
      // In thesis mode, ensure header is always hidden
      header.classList.add('nav-hidden');
    }
    lastScrollY = window.scrollY;

    // Add 'scrolled' class for background change
    if (window.scrollY > 50) {
      header.classList.add('scrolled');
    } else {
      header.classList.remove('scrolled');
    }
  });

  // Hamburger menu functionality
  const hamburgerMenu = document.getElementById('hamburger-menu');
  const menuContainers = document.getElementById('menu-containers');

  if (hamburgerMenu && menuContainers) {
    hamburgerMenu.addEventListener('click', () => {
      hamburgerMenu.classList.toggle('open');
      menuContainers.classList.toggle('open');
    });

    // Close menu when clicking outside
    document.addEventListener('click', (event) => {
      if (!hamburgerMenu.contains(event.target) && !menuContainers.contains(event.target)) {
        hamburgerMenu.classList.remove('open');
        menuContainers.classList.remove('open');
      }
    });
  }

  // Mode switcher functionality (Thesis Mode)
  const modeSwitcherCheckbox = document.getElementById('mode-switcher-checkbox');
  const modeSwitcherLabel = document.getElementById('mode-switcher-label');
  const body = document.body;
  const defaultNavLinks = document.querySelectorAll('.page-nav ul li .default-nav-link');
  const thesisNavLinks = document.querySelectorAll('.page-nav ul li .thesis-nav-link'); // Thesis links in hamburger menu
  const sectionNav = document.getElementById('section-nav');
  const defaultModeSectionNav = document.querySelector('.default-mode-nav');
  const thesisModeSectionNav = document.querySelector('.thesis-mode-nav');
  const mobileModeSwitcherCheckbox = document.getElementById('mobile-mode-switcher-checkbox');
  const mobileModeSwitcherLabel = document.getElementById('mobile-mode-switcher-label');

  const updateNavigationVisibility = (isInThesisMode) => {
    // Update hamburger menu links
    defaultNavLinks.forEach(link => {
      link.closest('li').style.display = isInThesisMode ? 'none' : 'block';
    });
    thesisNavLinks.forEach(link => {
      link.closest('li').style.display = isInThesisMode ? 'block' : 'none';
    });

    // Update bubble navigation visibility
    if (sectionNav) {
      if (isInThesisMode) {
        defaultModeSectionNav.style.display = 'none';
        thesisModeSectionNav.style.display = 'flex'; // Use flex to maintain layout
      } else {
        defaultModeSectionNav.style.display = 'flex'; // Use flex to maintain layout
        thesisModeSectionNav.style.display = 'none';
      }
    }

    // Hide header in thesis mode
    if (mainNav) {
      if (isInThesisMode) {
        mainNav.classList.add('hide-in-thesis-mode');
      } else {
        mainNav.classList.remove('hide-in-thesis-mode');
      }
    }
  };

  const toggleThesisMode = (isInThesisMode) => {
    body.classList.toggle('thesis-mode', isInThesisMode);
    if (modeSwitcherLabel) {
      modeSwitcherLabel.textContent = isInThesisMode ? 'Thesis Mode' : 'Default Mode';
    }
    if (mobileModeSwitcherLabel) {
      mobileModeSwitcherLabel.textContent = isInThesisMode ? 'Thesis Mode' : 'Default Mode';
    }

    updateNavigationVisibility(isInThesisMode);

    // Re-initialize Mermaid when switching to thesis mode or on initial load
    // Use a setTimeout to ensure DOM is fully rendered and styles are applied
    setTimeout(() => {
      mermaid.initialize({
        startOnLoad: true,
        theme: document.documentElement.getAttribute('data-theme') === 'dark' ? 'dark' : 'default',
        flowchart: {
          htmlLabels: true,
          curve: 'linear', // Use 'linear' or 'basis' for smoother curves
        },
        securityLevel: 'loose', // Allow more flexibility in diagram definitions
      });

      // Explicitly render all mermaid diagrams in both default and thesis sections
      document.querySelectorAll('.default-mode-section .mermaid, .thesis-mode-section .mermaid').forEach(mermaidElement => {
        // Check if the element has already been processed by Mermaid
        if (!mermaidElement.dataset.processed) {
          const diagramContent = mermaidElement.textContent.trim();
          const uniqueId = 'mermaid-svg-' + Math.random().toString(36).substr(2, 9);

          // Regex to capture the entire Mermaid definition, including optional config block
          // It looks for '--- config: ---' followed by content, or directly for a diagram type.
          // The 's' flag allows '.' to match newlines.
          const mermaidRegex = /(?:^---\s*config:[\s\S]*?---\s*\n)?\s*(graph|flowchart|sequenceDiagram|gantt|classDiagram|stateDiagram|erDiagram|journey|gitGraph|pie|C4Context|C4Container|C4Component|C4Dynamic|C4Deployment)[\s\S]*/s;
          const match = diagramContent.match(mermaidRegex);
          const cleanDiagramDefinition = match ? match[0].trim() : diagramContent;

          console.log('Attempting to render Mermaid diagram with content:', cleanDiagramDefinition); // Debugging line

          // Clear previous content before rendering
          mermaidElement.innerHTML = '';
          mermaidElement.removeAttribute('data-processed');

          mermaid.render(uniqueId, cleanDiagramDefinition).then(({ svg }) => {
            mermaidElement.innerHTML = svg;
            mermaidElement.dataset.processed = 'true'; // Mark as processed
          }).catch(error => {
            console.error('Mermaid rendering failed for element:', mermaidElement, 'Error:', error);
          });
        }
      });
    }, 100); // Small delay to ensure DOM is ready

    // Scroll to top on mode switch
    window.scrollTo({ top: 0, behavior: 'smooth' });
  };

  if (modeSwitcherCheckbox) {
    modeSwitcherCheckbox.addEventListener('change', () => {
      toggleThesisMode(modeSwitcherCheckbox.checked);
      if (mobileModeSwitcherCheckbox) {
        mobileModeSwitcherCheckbox.checked = modeSwitcherCheckbox.checked;
      }
    });
  }

  if (mobileModeSwitcherCheckbox) {
    mobileModeSwitcherCheckbox.addEventListener('change', () => {
      toggleThesisMode(mobileModeSwitcherCheckbox.checked);
      if (modeSwitcherCheckbox) {
        modeSwitcherCheckbox.checked = mobileModeSwitcherCheckbox.checked;
      }
    });
  }

  // Initial navigation setup based on current mode
  const initialIsInThesisMode = body.classList.contains('thesis-mode');
  if (modeSwitcherCheckbox) {
    modeSwitcherCheckbox.checked = initialIsInThesisMode;
  }
  if (mobileModeSwitcherCheckbox) {
    mobileModeSwitcherCheckbox.checked = initialIsInThesisMode;
  }
  if (modeSwitcherLabel) {
    modeSwitcherLabel.textContent = initialIsInThesisMode ? 'Thesis Mode' : 'Default Mode';
  }
  if (mobileModeSwitcherLabel) {
    mobileModeSwitcherLabel.textContent = initialIsInThesisMode ? 'Thesis Mode' : 'Default Mode';
  }
  updateNavigationVisibility(initialIsInThesisMode);

  // Smooth scroll for section-nav items
  if (sectionNav) {
    sectionNav.addEventListener('click', (event) => {
      const target = event.target.closest('.section-nav-item');
      if (target) {
        const targetId = target.dataset.target;
        const targetElement = document.getElementById(targetId);
        if (targetElement) {
          event.preventDefault();
          window.scrollTo({
            top: targetElement.offsetTop - mainNav.offsetHeight, // Adjust for fixed header
            behavior: 'smooth'
          });

          // Update active state for bubble navigation
          document.querySelectorAll('.section-nav-item').forEach(item => item.classList.remove('active'));
          target.classList.add('active');
        }
      }
    });

    // Update active state on scroll
      const sections = document.querySelectorAll('.content-container.full-screen');
      window.addEventListener('scroll', () => {
        let currentActive = '';
        sections.forEach(section => {
          const sectionTop = section.offsetTop - mainNav.offsetHeight;
          const sectionHeight = section.clientHeight;
          if (scrollY >= sectionTop && scrollY < sectionTop + sectionHeight) {
            currentActive = section.id;
          }
        });

        // Special handling for thesis sections to ensure correct active state
        if (body.classList.contains('thesis-mode')) {
          const thesisSections = document.querySelectorAll('.thesis-mode-section .content-container');
          thesisSections.forEach(section => {
            const sectionTop = section.offsetTop - mainNav.offsetHeight;
            const sectionHeight = section.clientHeight;
            if (scrollY >= sectionTop && scrollY < sectionTop + sectionHeight) {
              currentActive = section.id;
            }
          });
        }

      document.querySelectorAll('.section-nav-item').forEach(item => {
        item.classList.remove('active');
        if (item.dataset.target === currentActive) {
          item.classList.add('active');
        }
      });

      // Hide section-nav on the first section of both modes
      const firstDefaultSection = document.getElementById('hero');
      const firstThesisSection = document.getElementById('thesis-cover-page');
      const isInThesisMode = body.classList.contains('thesis-mode');

      if (sectionNav) {
        if (isInThesisMode) {
          if (currentActive === firstThesisSection.id) {
            sectionNav.style.display = 'none';
          } else {
            sectionNav.style.display = 'flex'; // Show as flex for vertical layout
          }
        } else {
          if (currentActive === firstDefaultSection.id) {
            sectionNav.style.display = 'none';
          } else {
            sectionNav.style.display = 'flex'; // Show as flex for vertical layout
          }
        }
      }
    });

    // Initial check for section-nav visibility on page load
    const initialSections = document.querySelectorAll('.content-container.full-screen, .faq-section'); // Include faq-section
    const initialCurrentActive = initialSections[0] ? initialSections[0].id : '';
    const initialIsInThesisMode = body.classList.contains('thesis-mode');
    const firstDefaultSection = document.getElementById('hero');
    const firstThesisSection = document.getElementById('thesis-cover-page');
    const thesisArchitectureSection = document.getElementById('thesis-architecture');
    const thesisImplementationSection = document.getElementById('thesis-implementation');

    if (sectionNav) {
      if (initialIsInThesisMode) {
        if (initialCurrentActive === firstThesisSection.id) {
          sectionNav.style.display = 'none';
        } else {
          sectionNav.style.display = 'flex';
        }
      } else {
        if (initialCurrentActive === firstDefaultSection.id) {
          sectionNav.style.display = 'none';
        } else {
          sectionNav.style.display = 'flex';
        }
      }
    }
  }

  // Function to copy code to clipboard
  window.copyCode = function(button) {
    const codeBlock = button.closest('.code-snippet-container').querySelector('code');
    const textToCopy = codeBlock.textContent;
    navigator.clipboard.writeText(textToCopy).then(() => {
      const originalText = button.textContent;
      button.textContent = 'Copied!';
      setTimeout(() => {
        button.textContent = originalText;
      }, 2000);
    }).catch(err => {
      console.error('Failed to copy code: ', err);
    });
  };

  // FAQ Accordion functionality for home.html
  const faqQuestions = document.querySelectorAll('.faq-section .faq-question');
  faqQuestions.forEach(question => {
    question.addEventListener('click', () => {
      const answer = question.nextElementSibling;
      const icon = question.querySelector('.faq-icon .material-symbols-outlined');

      answer.classList.toggle('hidden');
      if (answer.classList.contains('hidden')) {
        icon.textContent = 'add';
      } else {
        icon.textContent = 'remove';
      }
    });
  });

  // Chart.js for Thesis Methodology section
  const architectureChartCanvas = document.getElementById('architectureChart');
  if (architectureChartCanvas) {
    const ctx = architectureChartCanvas.getContext('2d');
    new Chart(ctx, {
      type: 'doughnut', // Changed to doughnut chart
      data: {
        labels: ['Frontend (WebRTC)', 'Backend (STT, MT, TTS)', 'Network Latency'],
        datasets: [{
          label: 'System Component Contribution',
          data: [25, 60, 15], // Example data in percentage
          backgroundColor: [
            'rgba(75, 192, 192, 0.6)',
            'rgba(153, 102, 255, 0.6)',
            'rgba(255, 159, 64, 0.6)'
          ],
          borderColor: [
            'rgba(75, 192, 192, 1)',
            'rgba(153, 102, 255, 1)',
            'rgba(255, 159, 64, 1)'
          ],
          borderWidth: 1
        }]
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        plugins: {
          legend: {
            position: 'top',
            labels: {
              color: 'var(--text-color)'
            }
          },
          title: {
            display: true,
            text: 'System Architecture Component Contribution',
            color: 'var(--text-color)',
            font: {
              size: 18
            }
          },
          tooltip: {
            callbacks: {
              label: function(context) {
                let label = context.label || '';
                if (label) {
                  label += ': ';
                }
                if (context.parsed !== null) {
                  label += context.parsed + '%';
                }
                return label;
              }
            }
          }
        }
      }
    });
  }

  // Chart.js for Thesis Results section
  const performanceChartCanvas = document.getElementById('performanceChart');
  if (performanceChartCanvas) {
    const ctx = performanceChartCanvas.getContext('2d');
    new Chart(ctx, {
      type: 'line', // Line chart for performance over time/metrics
      data: {
        labels: ['Initial Load', '50 Users', '100 Users', '150 Users', '200+ Users'],
        datasets: [{
          label: 'Average End-to-End Latency (ms)',
          data: [300, 400, 550, 700, 900], // Example data
          fill: false,
          borderColor: 'rgba(75, 192, 192, 1)',
          tension: 0.1
        }, {
          label: 'Voice Cloning Fidelity (MOS)',
          data: [4.5, 4.3, 4.1, 3.9, 3.7], // Example MOS score (1-5)
          fill: false,
          borderColor: 'rgba(153, 102, 255, 1)',
          tension: 0.1,
          yAxisID: 'y1' // Use a second Y-axis for MOS
        }]
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        interaction: {
          mode: 'index',
          intersect: false,
        },
        stacked: false,
        scales: {
          y: {
            type: 'linear',
            display: true,
            position: 'left',
            beginAtZero: true,
            title: {
              display: true,
              text: 'Latency (ms)',
              color: 'var(--text-color)'
            },
            ticks: {
              color: 'var(--secondary-color)'
            },
            grid: {
              color: 'rgba(var(--text-color-rgb), 0.1)'
            }
          },
          y1: {
            type: 'linear',
            display: true,
            position: 'right',
            beginAtZero: false,
            min: 1,
            max: 5,
            title: {
              display: true,
              text: 'MOS Score (1-5)',
              color: 'var(--text-color)'
            },
            ticks: {
              color: 'var(--secondary-color)'
            },
            grid: {
              drawOnChartArea: false, // Only draw grid lines for the first y-axis
              color: 'rgba(var(--text-color-rgb), 0.1)'
            }
          },
          x: {
            title: {
              display: true,
              text: 'Number of Simultaneous Users',
              color: 'var(--text-color)'
            },
            ticks: {
              color: 'var(--secondary-color)'
            },
            grid: {
              color: 'rgba(var(--text-color-rgb), 0.1)'
            }
          }
        },
        plugins: {
          legend: {
            labels: {
              color: 'var(--text-color)'
            }
          },
          title: {
            display: true,
            text: 'System Performance Under Load',
            color: 'var(--text-color)',
            font: {
              size: 18
            }
          }
        }
      }
    });
  }

  // Chart.js for Thesis Results section - Latency Breakdown Chart
  const latencyBreakdownChartCanvas = document.getElementById('latencyBreakdownChart');
  if (latencyBreakdownChartCanvas) {
    const ctx = latencyBreakdownChartCanvas.getContext('2d');
    new Chart(ctx, {
      type: 'bar', // Bar chart for latency breakdown
      data: {
        labels: ['STT Inference', 'MT Inference', 'TTS Inference (Piper)', 'TTS Inference (F5-TTS)', 'VAD Processing', 'Network Transmission'],
        datasets: [{
          label: 'Average Latency (ms)',
          data: [150, 100, 50, 200, 20, 80], // Example data in ms
          backgroundColor: [
            'rgba(255, 99, 132, 0.6)',
            'rgba(54, 162, 235, 0.6)',
            'rgba(255, 206, 86, 0.6)',
            'rgba(75, 192, 192, 0.6)',
            'rgba(153, 102, 255, 0.6)',
            'rgba(255, 159, 64, 0.6)'
          ],
          borderColor: [
            'rgba(255, 99, 132, 1)',
            'rgba(54, 162, 235, 1)',
            'rgba(255, 206, 86, 1)',
            'rgba(75, 192, 192, 1)',
            'rgba(153, 102, 255, 1)',
            'rgba(255, 159, 64, 1)'
          ],
          borderWidth: 1
        }]
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        plugins: {
          legend: {
            display: false,
            labels: {
              color: 'var(--text-color)'
            }
          },
          title: {
            display: true,
            text: 'Component-wise Latency Breakdown',
            color: 'var(--text-color)',
            font: {
              size: 18
            }
          },
          tooltip: {
            callbacks: {
              label: function(context) {
                let label = context.dataset.label || '';
                if (label) {
                  label += ': ';
                }
                if (context.parsed.y !== null) {
                  label += context.parsed.y + ' ms';
                }
                return label;
              }
            }
          }
        },
        scales: {
          y: {
            beginAtZero: true,
            title: {
              display: true,
              text: 'Latency (ms)',
              color: 'var(--text-color)'
            },
            ticks: {
              color: 'var(--secondary-color)'
            },
            grid: {
              color: 'rgba(var(--text-color-rgb), 0.1)'
            }
          },
          x: {
            ticks: {
              color: 'var(--secondary-color)'
            },
            grid: {
              color: 'rgba(var(--text-color-rgb), 0.1)'
            }
          }
        }
      }
    });
  }
});
