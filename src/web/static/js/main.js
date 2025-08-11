/**
 * Main JavaScript file for Smart Parking System
 */

// Global application state
window.ParkingApp = {
    initialized: false,
    currentPage: '',
    wsManager: null,
    apiClient: null,
    config: {
        refreshInterval: 30000,
        chartUpdateInterval: 60000,
        notificationTimeout: 5000
    }
};

/**
 * Initialize the application
 */
function initializeApp() {
    if (window.ParkingApp.initialized) return;
    
    console.log('Initializing Smart Parking System...');
    
    // Set current page
    window.ParkingApp.currentPage = getCurrentPage();
    
    // Initialize global components
    initializeGlobalComponents();
    
    // Initialize page-specific components
    initializePageComponents();
    
    // Setup global event listeners
    setupGlobalEventListeners();
    
    // Mark as initialized
    window.ParkingApp.initialized = true;
    
    console.log('Smart Parking System initialized successfully');
}

/**
 * Get current page from URL
 */
function getCurrentPage() {
    const path = window.location.pathname;
    if (path === '/' || path === '/dashboard') return 'dashboard';
    if (path.startsWith('/vehicles')) return 'vehicles';
    if (path.startsWith('/parking')) return 'parking';
    if (path.startsWith('/analytics')) return 'analytics';
    if (path.startsWith('/cameras')) return 'cameras';
    if (path.startsWith('/settings')) return 'settings';
    return 'unknown';
}

/**
 * Initialize global components
 */
function initializeGlobalComponents() {
    // Initialize API client
    window.ParkingApp.apiClient = window.apiClient;
    
    // Initialize WebSocket manager (will be available after websocket.js loads)
    if (window.wsManager) {
        window.ParkingApp.wsManager = window.wsManager;
        setupWebSocketHandlers();
    }
    
    // Initialize Bootstrap components
    initializeBootstrapComponents();
    
    // Setup navigation
    setupNavigation();
    
    // Setup search functionality
    setupGlobalSearch();
}

/**
 * Initialize page-specific components
 */
function initializePageComponents() {
    const page = window.ParkingApp.currentPage;
    
    switch (page) {
        case 'dashboard':
            if (typeof initializeDashboard === 'function') {
                initializeDashboard();
            }
            break;
        case 'vehicles':
            if (typeof initializeVehiclesPage === 'function') {
                initializeVehiclesPage();
            }
            break;
        case 'parking':
            if (typeof initializeParkingMap === 'function') {
                initializeParkingMap();
            }
            break;
        case 'analytics':
            if (typeof initializeAnalytics === 'function') {
                initializeAnalytics();
            }
            break;
        case 'cameras':
            if (typeof initializeCamerasPage === 'function') {
                initializeCamerasPage();
            }
            break;
        default:
            console.log(`No specific initialization for page: ${page}`);
    }
}

/**
 * Initialize Bootstrap components
 */
function initializeBootstrapComponents() {
    // Initialize tooltips
    const tooltipTriggerList = [].slice.call(document.querySelectorAll('[data-bs-toggle="tooltip"]'));
    tooltipTriggerList.map(function (tooltipTriggerEl) {
        return new bootstrap.Tooltip(tooltipTriggerEl);
    });
    
    // Initialize popovers
    const popoverTriggerList = [].slice.call(document.querySelectorAll('[data-bs-toggle="popover"]'));
    popoverTriggerList.map(function (popoverTriggerEl) {
        return new bootstrap.Popover(popoverTriggerEl);
    });
    
    // Initialize modals
    const modalElements = document.querySelectorAll('.modal');
    modalElements.forEach(modalEl => {
        new bootstrap.Modal(modalEl);
    });
}

/**
 * Setup navigation
 */
function setupNavigation() {
    // Highlight current page in navigation
    const currentPage = window.ParkingApp.currentPage;
    const navLinks = document.querySelectorAll('.navbar-nav .nav-link');
    
    navLinks.forEach(link => {
        const href = link.getAttribute('href');
        if (href === '/' && currentPage === 'dashboard') {
            link.classList.add('active');
        } else if (href && href.includes(currentPage)) {
            link.classList.add('active');
        }
    });
    
    // Setup mobile menu toggle
    const navbarToggler = document.querySelector('.navbar-toggler');
    const navbarCollapse = document.querySelector('.navbar-collapse');
    
    if (navbarToggler && navbarCollapse) {
        navbarToggler.addEventListener('click', function() {
            navbarCollapse.classList.toggle('show');
        });
    }
}

/**
 * Setup global search functionality
 */
function setupGlobalSearch() {
    // Add search functionality to navigation if needed
    const searchInput = document.getElementById('global-search');
    if (searchInput) {
        searchInput.addEventListener('input', debounce(function(e) {
            const query = e.target.value.trim();
            if (query.length >= 2) {
                performGlobalSearch(query);
            }
        }, 300));
    }
}

/**
 * Perform global search
 */
async function performGlobalSearch(query) {
    try {
        const results = await window.apiClient.get(`/vehicles/search/${encodeURIComponent(query)}`);
        displaySearchResults(results.vehicles || []);
    } catch (error) {
        console.error('Search error:', error);
    }
}

/**
 * Display search results
 */
function displaySearchResults(results) {
    // Implementation depends on UI design
    console.log('Search results:', results);
}

/**
 * Setup WebSocket event handlers
 */
function setupWebSocketHandlers() {
    const wsManager = window.ParkingApp.wsManager;
    if (!wsManager) return;
    
    // Global WebSocket event handlers
    wsManager.on('connected', function() {
        console.log('WebSocket connected');
        updateConnectionStatus(true);
    });
    
    wsManager.on('disconnected', function() {
        console.log('WebSocket disconnected');
        updateConnectionStatus(false);
    });
    
    wsManager.on('error', function(error) {
        console.error('WebSocket error:', error);
        showNotification('Connection Error', 'WebSocket connection error', 'danger');
    });
    
    // System alerts
    wsManager.on('system_alert', function(data) {
        showNotification('System Alert', data.message, data.level || 'warning');
    });
    
    // Camera status updates
    wsManager.on('camera_status', function(data) {
        updateCameraStatus(data);
    });
}

/**
 * Update connection status indicator
 */
function updateConnectionStatus(connected) {
    const statusElement = document.getElementById('connection-status');
    if (statusElement) {
        const badge = statusElement.querySelector('.badge');
        if (badge) {
            if (connected) {
                badge.className = 'badge bg-success';
                badge.innerHTML = '<i class="fas fa-wifi"></i> Connected';
            } else {
                badge.className = 'badge bg-danger';
                badge.innerHTML = '<i class="fas fa-wifi"></i> Disconnected';
            }
        }
    }
}

/**
 * Update camera status
 */
function updateCameraStatus(data) {
    // Update camera status indicators if on cameras page
    if (window.ParkingApp.currentPage === 'cameras' && typeof updateCameraStatusDisplay === 'function') {
        updateCameraStatusDisplay(data);
    }
}

/**
 * Setup global event listeners
 */
function setupGlobalEventListeners() {
    // Handle page visibility change
    document.addEventListener('visibilitychange', function() {
        if (document.visibilityState === 'visible') {
            // Page became visible, refresh data
            refreshCurrentPageData();
        }
    });
    
    // Handle window resize
    window.addEventListener('resize', debounce(function() {
        handleWindowResize();
    }, 250));
    
    // Handle keyboard shortcuts
    document.addEventListener('keydown', function(e) {
        handleKeyboardShortcuts(e);
    });
    
    // Handle form submissions
    document.addEventListener('submit', function(e) {
        handleFormSubmission(e);
    });
}

/**
 * Refresh current page data
 */
function refreshCurrentPageData() {
    const page = window.ParkingApp.currentPage;
    
    switch (page) {
        case 'dashboard':
            if (typeof loadParkingStatus === 'function') {
                loadParkingStatus();
            }
            break;
        case 'vehicles':
            if (typeof loadVehicles === 'function') {
                loadVehicles();
            }
            break;
        case 'parking':
            if (typeof loadParkingMap === 'function') {
                loadParkingMap();
            }
            break;
        case 'analytics':
            if (typeof loadAnalyticsData === 'function') {
                loadAnalyticsData();
            }
            break;
    }
}

/**
 * Handle window resize
 */
function handleWindowResize() {
    // Resize charts if they exist
    if (window.Chart) {
        Chart.helpers.each(Chart.instances, function(instance) {
            instance.resize();
        });
    }
    
    // Update parking map if visible
    if (window.ParkingApp.currentPage === 'parking' && typeof resizeParkingMap === 'function') {
        resizeParkingMap();
    }
}

/**
 * Handle keyboard shortcuts
 */
function handleKeyboardShortcuts(e) {
    // Ctrl/Cmd + R: Refresh current page data
    if ((e.ctrlKey || e.metaKey) && e.key === 'r') {
        e.preventDefault();
        refreshCurrentPageData();
        showNotification('Refreshed', 'Page data refreshed', 'info');
    }
    
    // Escape: Close modals/overlays
    if (e.key === 'Escape') {
        const modals = document.querySelectorAll('.modal.show');
        modals.forEach(modal => {
            const bsModal = bootstrap.Modal.getInstance(modal);
            if (bsModal) bsModal.hide();
        });
    }
}

/**
 * Handle form submissions
 */
function handleFormSubmission(e) {
    const form = e.target;
    if (!form.matches('form')) return;
    
    // Add loading state to submit button
    const submitBtn = form.querySelector('button[type="submit"]');
    if (submitBtn) {
        submitBtn.disabled = true;
        const originalText = submitBtn.innerHTML;
        submitBtn.innerHTML = '<i class="fas fa-spinner fa-spin"></i> Processing...';
        
        // Restore button after 5 seconds (fallback)
        setTimeout(() => {
            submitBtn.disabled = false;
            submitBtn.innerHTML = originalText;
        }, 5000);
    }
}

/**
 * Application error handler
 */
window.addEventListener('error', function(e) {
    console.error('Application error:', e.error);
    
    // Show user-friendly error message
    showNotification(
        'Application Error', 
        'An unexpected error occurred. Please refresh the page.', 
        'danger'
    );
});

/**
 * Unhandled promise rejection handler
 */
window.addEventListener('unhandledrejection', function(e) {
    console.error('Unhandled promise rejection:', e.reason);
    
    // Show user-friendly error message
    showNotification(
        'Network Error', 
        'A network error occurred. Please check your connection.', 
        'warning'
    );
});

// Initialize app when DOM is loaded
document.addEventListener('DOMContentLoaded', function() {
    // Small delay to ensure all scripts are loaded
    setTimeout(initializeApp, 100);
});

// Initialize app when window is loaded (fallback)
window.addEventListener('load', function() {
    if (!window.ParkingApp.initialized) {
        initializeApp();
    }
});
