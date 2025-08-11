/**
 * Utility functions for Smart Parking System
 */

// API Base URL
const API_BASE_URL = '/api/v1';

/**
 * HTTP Client for API calls
 */
class APIClient {
    constructor(baseURL = API_BASE_URL) {
        this.baseURL = baseURL;
    }
    
    async request(endpoint, options = {}) {
        const url = `${this.baseURL}${endpoint}`;
        const config = {
            headers: {
                'Content-Type': 'application/json',
                ...options.headers
            },
            ...options
        };
        
        try {
            showLoading(true);
            const response = await fetch(url, config);
            
            if (!response.ok) {
                throw new Error(`HTTP error! status: ${response.status}`);
            }
            
            const data = await response.json();
            return data;
        } catch (error) {
            console.error('API request failed:', error);
            showNotification('Error', `API request failed: ${error.message}`, 'danger');
            throw error;
        } finally {
            showLoading(false);
        }
    }
    
    async get(endpoint) {
        return this.request(endpoint, { method: 'GET' });
    }
    
    async post(endpoint, data) {
        return this.request(endpoint, {
            method: 'POST',
            body: JSON.stringify(data)
        });
    }
    
    async put(endpoint, data) {
        return this.request(endpoint, {
            method: 'PUT',
            body: JSON.stringify(data)
        });
    }
    
    async delete(endpoint) {
        return this.request(endpoint, { method: 'DELETE' });
    }
}

// Global API client instance
const apiClient = new APIClient();

/**
 * Utility Functions
 */

// Show/hide loading overlay
function showLoading(show = true) {
    const overlay = document.getElementById('loading-overlay');
    if (overlay) {
        if (show) {
            overlay.classList.remove('d-none');
        } else {
            overlay.classList.add('d-none');
        }
    }
}

// Show notification toast
function showNotification(title, message, type = 'info') {
    const toast = document.getElementById('notification-toast');
    if (toast) {
        const toastHeader = toast.querySelector('.toast-header strong');
        const toastBody = toast.querySelector('.toast-body');
        const toastIcon = toast.querySelector('.toast-header i');
        
        if (toastHeader) toastHeader.textContent = title;
        if (toastBody) toastBody.textContent = message;
        
        // Update icon based on type
        if (toastIcon) {
            toastIcon.className = `fas me-2 ${getIconForType(type)}`;
        }
        
        // Update toast styling
        toast.className = `toast border-${type}`;
        
        const bsToast = new bootstrap.Toast(toast);
        bsToast.show();
    }
}

// Get icon class for notification type
function getIconForType(type) {
    const icons = {
        'success': 'fa-check-circle text-success',
        'info': 'fa-info-circle text-info',
        'warning': 'fa-exclamation-triangle text-warning',
        'danger': 'fa-exclamation-circle text-danger',
        'error': 'fa-exclamation-circle text-danger'
    };
    return icons[type] || 'fa-bell text-primary';
}

// Format date/time
function formatDateTime(dateString, options = {}) {
    const date = new Date(dateString);
    const defaultOptions = {
        year: 'numeric',
        month: '2-digit',
        day: '2-digit',
        hour: '2-digit',
        minute: '2-digit',
        second: '2-digit'
    };
    
    return date.toLocaleString('vi-VN', { ...defaultOptions, ...options });
}

// Format duration in minutes to human readable
function formatDuration(minutes) {
    if (minutes < 60) {
        return `${Math.round(minutes)} phút`;
    } else if (minutes < 1440) {
        const hours = Math.floor(minutes / 60);
        const mins = Math.round(minutes % 60);
        return `${hours} giờ ${mins} phút`;
    } else {
        const days = Math.floor(minutes / 1440);
        const hours = Math.floor((minutes % 1440) / 60);
        return `${days} ngày ${hours} giờ`;
    }
}

// Format license plate for display
function formatLicensePlate(plate) {
    if (!plate) return '';
    
    // Vietnamese license plate format: 51A-123.45
    const match = plate.match(/^(\d{2})([A-Z]+)-?(\d{3})\.?(\d{2})$/);
    if (match) {
        return `${match[1]}${match[2]}-${match[3]}.${match[4]}`;
    }
    
    return plate.toUpperCase();
}

// Get vehicle type icon
function getVehicleTypeIcon(type) {
    const icons = {
        'car': 'fas fa-car',
        'motorcycle': 'fas fa-motorcycle',
        'bus': 'fas fa-bus',
        'truck': 'fas fa-truck'
    };
    return icons[type] || 'fas fa-car';
}

// Get vehicle type color
function getVehicleTypeColor(type) {
    const colors = {
        'car': 'primary',
        'motorcycle': 'success',
        'bus': 'warning',
        'truck': 'info'
    };
    return colors[type] || 'secondary';
}

// Update current time display
function updateCurrentTime() {
    const timeElement = document.getElementById('current-time');
    if (timeElement) {
        const now = new Date();
        timeElement.textContent = formatDateTime(now.toISOString(), {
            hour: '2-digit',
            minute: '2-digit',
            second: '2-digit'
        });
    }
}

// Start time update interval
function startTimeUpdate() {
    updateCurrentTime();
    setInterval(updateCurrentTime, 1000);
}

// Debounce function
function debounce(func, wait, immediate) {
    let timeout;
    return function executedFunction(...args) {
        const later = () => {
            timeout = null;
            if (!immediate) func(...args);
        };
        const callNow = immediate && !timeout;
        clearTimeout(timeout);
        timeout = setTimeout(later, wait);
        if (callNow) func(...args);
    };
}

// Throttle function
function throttle(func, limit) {
    let inThrottle;
    return function(...args) {
        if (!inThrottle) {
            func.apply(this, args);
            inThrottle = true;
            setTimeout(() => inThrottle = false, limit);
        }
    };
}

// Copy to clipboard
async function copyToClipboard(text) {
    try {
        await navigator.clipboard.writeText(text);
        showNotification('Copied', 'Text copied to clipboard', 'success');
    } catch (error) {
        console.error('Failed to copy to clipboard:', error);
        showNotification('Error', 'Failed to copy to clipboard', 'danger');
    }
}

// Download data as JSON
function downloadJSON(data, filename = 'data.json') {
    const blob = new Blob([JSON.stringify(data, null, 2)], {
        type: 'application/json'
    });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = filename;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    URL.revokeObjectURL(url);
}

// Download data as CSV
function downloadCSV(data, filename = 'data.csv') {
    if (!data || data.length === 0) return;
    
    const headers = Object.keys(data[0]);
    const csvContent = [
        headers.join(','),
        ...data.map(row => headers.map(header => {
            const value = row[header];
            return typeof value === 'string' && value.includes(',') 
                ? `"${value}"` 
                : value;
        }).join(','))
    ].join('\n');
    
    const blob = new Blob([csvContent], { type: 'text/csv' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = filename;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    URL.revokeObjectURL(url);
}

// Validate license plate format
function validateLicensePlate(plate) {
    const patterns = [
        /^\d{2}[A-Z]-\d{3}\.\d{2}$/,  // 51A-123.45
        /^\d{2}[A-Z]{2}-\d{3}\.\d{2}$/, // 51AB-123.45
        /^\d{2}[A-Z]-\d{4}$/,         // 51A-1234
        /^\d{2}[A-Z]\d{3}\.\d{2}$/,   // 51A123.45
        /^\d{2}[A-Z]\d{4}$/           // 51A1234
    ];
    
    return patterns.some(pattern => pattern.test(plate));
}

// Format number with thousands separator
function formatNumber(num) {
    return new Intl.NumberFormat('vi-VN').format(num);
}

// Calculate percentage
function calculatePercentage(value, total) {
    if (total === 0) return 0;
    return Math.round((value / total) * 100);
}

// Get status badge HTML
function getStatusBadge(status, text) {
    const colors = {
        'active': 'success',
        'inactive': 'secondary',
        'online': 'success',
        'offline': 'danger',
        'warning': 'warning',
        'error': 'danger'
    };
    
    const color = colors[status] || 'secondary';
    return `<span class="badge bg-${color}">${text || status}</span>`;
}

// Initialize tooltips
function initializeTooltips() {
    const tooltipTriggerList = [].slice.call(document.querySelectorAll('[data-bs-toggle="tooltip"]'));
    tooltipTriggerList.map(function (tooltipTriggerEl) {
        return new bootstrap.Tooltip(tooltipTriggerEl);
    });
}

// Initialize popovers
function initializePopovers() {
    const popoverTriggerList = [].slice.call(document.querySelectorAll('[data-bs-toggle="popover"]'));
    popoverTriggerList.map(function (popoverTriggerEl) {
        return new bootstrap.Popover(popoverTriggerEl);
    });
}

// Initialize page
function initializePage() {
    startTimeUpdate();
    initializeTooltips();
    initializePopovers();
}

// DOM ready
document.addEventListener('DOMContentLoaded', function() {
    initializePage();
});

// Export for module usage
if (typeof module !== 'undefined' && module.exports) {
    module.exports = {
        APIClient,
        apiClient,
        showLoading,
        showNotification,
        formatDateTime,
        formatDuration,
        formatLicensePlate,
        getVehicleTypeIcon,
        getVehicleTypeColor,
        debounce,
        throttle,
        copyToClipboard,
        downloadJSON,
        downloadCSV,
        validateLicensePlate,
        formatNumber,
        calculatePercentage,
        getStatusBadge
    };
}
