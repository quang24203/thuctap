/**
 * Dashboard JavaScript for Smart Parking System
 */

// Global variables
let occupancyChart = null;
let recentVehicles = [];
let currentStats = {
    total_slots: 300,
    occupied_slots: 0,
    available_slots: 300,
    occupancy_rate: 0
};

/**
 * Initialize Dashboard
 */
function initializeDashboard() {
    loadParkingStatus();
    loadRecentVehicles();
    loadZoneStatus();
    initializeOccupancyChart();
    
    // Setup auto-refresh
    setInterval(loadParkingStatus, 30000); // Refresh every 30 seconds
    setInterval(loadRecentVehicles, 60000); // Refresh every minute
}

/**
 * Load parking status from API
 */
async function loadParkingStatus() {
    try {
        const data = await apiClient.get('/parking/status');
        updateParkingStats(data);
    } catch (error) {
        console.error('Error loading parking status:', error);
    }
}

/**
 * Update parking statistics display
 */
function updateParkingStats(data) {
    currentStats = data;
    
    // Update stat cards
    const totalSlotsEl = document.getElementById('total-slots');
    const availableSlotsEl = document.getElementById('available-slots');
    const occupiedSlotsEl = document.getElementById('occupied-slots');
    const occupancyRateEl = document.getElementById('occupancy-rate');
    const occupancyProgressEl = document.getElementById('occupancy-progress');
    
    if (totalSlotsEl) totalSlotsEl.textContent = formatNumber(data.total_slots);
    if (availableSlotsEl) availableSlotsEl.textContent = formatNumber(data.available_slots);
    if (occupiedSlotsEl) occupiedSlotsEl.textContent = formatNumber(data.occupied_slots);
    
    if (occupancyRateEl) {
        occupancyRateEl.textContent = `${Math.round(data.occupancy_rate)}%`;
    }
    
    if (occupancyProgressEl) {
        occupancyProgressEl.style.width = `${data.occupancy_rate}%`;
        occupancyProgressEl.setAttribute('aria-valuenow', data.occupancy_rate);
    }
    
    // Update chart if it exists
    if (occupancyChart) {
        updateOccupancyChart();
    }
}

/**
 * Load recent vehicles
 */
async function loadRecentVehicles() {
    try {
        const data = await apiClient.get('/vehicles?active_only=false&limit=10');
        recentVehicles = data.vehicles || [];
        updateRecentVehiclesTable();
    } catch (error) {
        console.error('Error loading recent vehicles:', error);
    }
}

/**
 * Update recent vehicles table
 */
function updateRecentVehiclesTable() {
    const tableBody = document.querySelector('#recent-vehicles-table tbody');
    if (!tableBody) return;
    
    tableBody.innerHTML = '';
    
    recentVehicles.slice(0, 10).forEach(vehicle => {
        const row = document.createElement('tr');
        row.className = 'fade-in';
        
        const action = vehicle.is_active ? 'Entry' : 'Exit';
        const actionClass = vehicle.is_active ? 'success' : 'info';
        const time = vehicle.is_active ? vehicle.entry_time : vehicle.exit_time;
        
        row.innerHTML = `
            <td>
                <strong>${formatLicensePlate(vehicle.license_plate)}</strong>
            </td>
            <td>
                <i class="${getVehicleTypeIcon(vehicle.vehicle_type)} text-${getVehicleTypeColor(vehicle.vehicle_type)}"></i>
                ${vehicle.vehicle_type}
            </td>
            <td>
                <span class="badge bg-${actionClass}">${action}</span>
            </td>
            <td>
                <small>${formatDateTime(time, { 
                    hour: '2-digit', 
                    minute: '2-digit' 
                })}</small>
            </td>
        `;
        
        tableBody.appendChild(row);
    });
    
    if (recentVehicles.length === 0) {
        const row = document.createElement('tr');
        row.innerHTML = `
            <td colspan="4" class="text-center text-muted">
                <i class="fas fa-info-circle"></i>
                No recent vehicle activity
            </td>
        `;
        tableBody.appendChild(row);
    }
}

/**
 * Add new vehicle to recent list (called from WebSocket)
 */
function addRecentVehicle(vehicleData, action) {
    const vehicle = {
        ...vehicleData,
        is_active: action === 'entry'
    };
    
    // Add to beginning of list
    recentVehicles.unshift(vehicle);
    
    // Keep only last 20 vehicles
    recentVehicles = recentVehicles.slice(0, 20);
    
    // Update table
    updateRecentVehiclesTable();
}

/**
 * Load zone status
 */
async function loadZoneStatus() {
    try {
        const data = await apiClient.get('/parking/zones');
        updateZoneStatus(data.zones || []);
    } catch (error) {
        console.error('Error loading zone status:', error);
    }
}

/**
 * Update zone status display
 */
function updateZoneStatus(zones) {
    const container = document.getElementById('zone-status-container');
    if (!container) return;
    
    container.innerHTML = '';
    
    zones.forEach(zone => {
        const zoneElement = document.createElement('div');
        zoneElement.className = 'mb-3';
        
        const occupancyRate = zone.total > 0 ? (zone.occupied / zone.total * 100) : 0;
        const progressColor = occupancyRate > 80 ? 'danger' : occupancyRate > 60 ? 'warning' : 'success';
        
        zoneElement.innerHTML = `
            <div class="d-flex justify-content-between align-items-center mb-1">
                <h6 class="mb-0">${zone.name}</h6>
                <small class="text-muted">${zone.occupied}/${zone.total}</small>
            </div>
            <div class="progress mb-1" style="height: 8px;">
                <div class="progress-bar bg-${progressColor}" 
                     style="width: ${occupancyRate}%"
                     aria-valuenow="${occupancyRate}" 
                     aria-valuemin="0" 
                     aria-valuemax="100">
                </div>
            </div>
            <small class="text-muted">${Math.round(occupancyRate)}% occupied</small>
        `;
        
        container.appendChild(zoneElement);
    });
}

/**
 * Initialize occupancy chart
 */
function initializeOccupancyChart() {
    const ctx = document.getElementById('occupancy-chart');
    if (!ctx) return;
    
    occupancyChart = new Chart(ctx, {
        type: 'line',
        data: {
            labels: [],
            datasets: [{
                label: 'Occupancy Rate (%)',
                data: [],
                borderColor: 'rgb(75, 192, 192)',
                backgroundColor: 'rgba(75, 192, 192, 0.1)',
                tension: 0.4,
                fill: true
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                legend: {
                    display: false
                }
            },
            scales: {
                y: {
                    beginAtZero: true,
                    max: 100,
                    ticks: {
                        callback: function(value) {
                            return value + '%';
                        }
                    }
                },
                x: {
                    display: true
                }
            },
            elements: {
                point: {
                    radius: 3,
                    hoverRadius: 6
                }
            }
        }
    });
    
    // Load initial data
    loadOccupancyTrends();
}

/**
 * Load occupancy trends
 */
async function loadOccupancyTrends() {
    try {
        const data = await apiClient.get('/statistics/occupancy?days=1');
        updateOccupancyChart(data.trends || []);
    } catch (error) {
        console.error('Error loading occupancy trends:', error);
    }
}

/**
 * Update occupancy chart
 */
function updateOccupancyChart(trends = null) {
    if (!occupancyChart) return;
    
    if (trends) {
        // Update with new data
        const labels = trends.map(trend => {
            const date = new Date(trend.timestamp);
            return date.toLocaleTimeString('vi-VN', { 
                hour: '2-digit', 
                minute: '2-digit' 
            });
        });
        
        const data = trends.map(trend => trend.occupancy_rate);
        
        occupancyChart.data.labels = labels;
        occupancyChart.data.datasets[0].data = data;
    } else {
        // Add current data point
        const now = new Date();
        const timeLabel = now.toLocaleTimeString('vi-VN', { 
            hour: '2-digit', 
            minute: '2-digit' 
        });
        
        occupancyChart.data.labels.push(timeLabel);
        occupancyChart.data.datasets[0].data.push(currentStats.occupancy_rate);
        
        // Keep only last 24 data points
        if (occupancyChart.data.labels.length > 24) {
            occupancyChart.data.labels.shift();
            occupancyChart.data.datasets[0].data.shift();
        }
    }
    
    occupancyChart.update('none');
}

/**
 * Refresh occupancy chart
 */
function refreshOccupancyChart() {
    loadOccupancyTrends();
}

/**
 * Export chart data
 */
function exportChart(chartType) {
    if (chartType === 'occupancy' && occupancyChart) {
        const chartData = {
            labels: occupancyChart.data.labels,
            data: occupancyChart.data.datasets[0].data,
            exported_at: new Date().toISOString()
        };
        
        downloadJSON(chartData, `occupancy_chart_${new Date().toISOString().split('T')[0]}.json`);
    }
}

/**
 * Start camera feed (placeholder)
 */
function startCameraFeed() {
    const feedContainer = document.getElementById('camera-feed');
    if (feedContainer) {
        feedContainer.innerHTML = `
            <div class="d-flex align-items-center justify-content-center h-100">
                <div class="text-center text-white">
                    <div class="spinner-border mb-3" role="status">
                        <span class="visually-hidden">Loading...</span>
                    </div>
                    <p>Connecting to camera feed...</p>
                    <small>This is a demo placeholder</small>
                </div>
            </div>
        `;
        
        // Simulate connection delay
        setTimeout(() => {
            feedContainer.innerHTML = `
                <div class="d-flex align-items-center justify-content-center h-100">
                    <div class="text-center text-white">
                        <i class="fas fa-video fa-3x mb-3"></i>
                        <p>Live camera feed would appear here</p>
                        <small>Integration with camera system required</small>
                    </div>
                </div>
            `;
        }, 2000);
    }
}

/**
 * Handle camera selector change
 */
function handleCameraChange() {
    const selector = document.getElementById('camera-selector');
    if (selector) {
        selector.addEventListener('change', function() {
            const selectedCamera = this.value;
            console.log('Selected camera:', selectedCamera);
            
            // Reset camera feed
            const feedContainer = document.getElementById('camera-feed');
            if (feedContainer) {
                feedContainer.innerHTML = `
                    <div class="d-flex align-items-center justify-content-center h-100">
                        <div class="text-center text-white">
                            <i class="fas fa-video fa-3x mb-3"></i>
                            <p>Camera feed: ${this.options[this.selectedIndex].text}</p>
                            <button class="btn btn-primary btn-sm" onclick="startCameraFeed()">
                                <i class="fas fa-play"></i>
                                Start Feed
                            </button>
                        </div>
                    </div>
                `;
            }
        });
    }
}

// Initialize dashboard when DOM is loaded
document.addEventListener('DOMContentLoaded', function() {
    initializeDashboard();
    handleCameraChange();
    
    // Setup WebSocket event handlers if available
    if (window.wsManager) {
        wsManager.on('parking_status_update', updateParkingStats);
        wsManager.on('vehicle_entry', (data) => addRecentVehicle(data, 'entry'));
        wsManager.on('vehicle_exit', (data) => addRecentVehicle(data, 'exit'));
    }
});
