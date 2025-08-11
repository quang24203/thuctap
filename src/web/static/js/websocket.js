/**
 * WebSocket Manager for Smart Parking System
 */

class WebSocketManager {
    constructor() {
        this.socket = null;
        this.reconnectAttempts = 0;
        this.maxReconnectAttempts = 5;
        this.reconnectInterval = 5000;
        this.isConnected = false;
        this.subscriptions = new Set();
        this.eventHandlers = new Map();
        
        this.init();
    }
    
    init() {
        this.connect();
        this.setupEventHandlers();
    }
    
    connect() {
        try {
            const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
            const wsUrl = `${protocol}//${window.location.host}/ws`;
            
            this.socket = new WebSocket(wsUrl);
            
            this.socket.onopen = (event) => {
                console.log('WebSocket connected');
                this.isConnected = true;
                this.reconnectAttempts = 0;
                this.updateConnectionStatus(true);
                
                // Subscribe to default events
                this.subscribe(['parking_status_update', 'vehicle_entry', 'vehicle_exit']);
                
                // Trigger connected event
                this.emit('connected', event);
            };
            
            this.socket.onmessage = (event) => {
                try {
                    const data = JSON.parse(event.data);
                    this.handleMessage(data);
                } catch (error) {
                    console.error('Error parsing WebSocket message:', error);
                }
            };
            
            this.socket.onclose = (event) => {
                console.log('WebSocket disconnected');
                this.isConnected = false;
                this.updateConnectionStatus(false);
                
                // Trigger disconnected event
                this.emit('disconnected', event);
                
                // Attempt to reconnect
                this.attemptReconnect();
            };
            
            this.socket.onerror = (error) => {
                console.error('WebSocket error:', error);
                this.emit('error', error);
            };
            
        } catch (error) {
            console.error('Error creating WebSocket connection:', error);
            this.attemptReconnect();
        }
    }
    
    attemptReconnect() {
        if (this.reconnectAttempts < this.maxReconnectAttempts) {
            this.reconnectAttempts++;
            console.log(`Attempting to reconnect... (${this.reconnectAttempts}/${this.maxReconnectAttempts})`);
            
            setTimeout(() => {
                this.connect();
            }, this.reconnectInterval);
        } else {
            console.error('Max reconnection attempts reached');
            this.updateConnectionStatus(false, 'Connection failed');
        }
    }
    
    handleMessage(data) {
        const { type, data: messageData, timestamp } = data;
        
        console.log('WebSocket message received:', type, messageData);
        
        // Emit event to registered handlers
        this.emit(type, messageData, timestamp);
        
        // Handle specific message types
        switch (type) {
            case 'parking_status_update':
                this.handleParkingStatusUpdate(messageData);
                break;
            case 'vehicle_entry':
                this.handleVehicleEntry(messageData);
                break;
            case 'vehicle_exit':
                this.handleVehicleExit(messageData);
                break;
            case 'detection_result':
                this.handleDetectionResult(messageData);
                break;
            case 'system_alert':
                this.handleSystemAlert(messageData);
                break;
            case 'subscription_confirmed':
                console.log('Subscriptions confirmed:', messageData.subscriptions);
                break;
            case 'pong':
                console.log('Pong received');
                break;
            default:
                console.log('Unknown message type:', type);
        }
    }
    
    handleParkingStatusUpdate(data) {
        // Update dashboard statistics
        if (typeof updateParkingStats === 'function') {
            updateParkingStats(data);
        }
        
        // Update parking map if visible
        if (typeof updateParkingMap === 'function') {
            updateParkingMap(data);
        }
    }
    
    handleVehicleEntry(data) {
        // Show notification
        this.showNotification('Vehicle Entry', `${data.license_plate} entered the parking lot`, 'success');
        
        // Update recent vehicles list
        if (typeof addRecentVehicle === 'function') {
            addRecentVehicle(data, 'entry');
        }
    }
    
    handleVehicleExit(data) {
        // Show notification
        this.showNotification('Vehicle Exit', `${data.license_plate} exited the parking lot`, 'info');
        
        // Update recent vehicles list
        if (typeof addRecentVehicle === 'function') {
            addRecentVehicle(data, 'exit');
        }
    }
    
    handleDetectionResult(data) {
        // Update detection statistics
        if (typeof updateDetectionStats === 'function') {
            updateDetectionStats(data);
        }
    }
    
    handleSystemAlert(data) {
        // Show system alert
        this.showNotification('System Alert', data.message, data.level || 'warning');
    }
    
    send(message) {
        if (this.isConnected && this.socket.readyState === WebSocket.OPEN) {
            this.socket.send(JSON.stringify(message));
        } else {
            console.warn('WebSocket not connected, message not sent:', message);
        }
    }
    
    subscribe(events) {
        if (!Array.isArray(events)) {
            events = [events];
        }
        
        events.forEach(event => this.subscriptions.add(event));
        
        this.send({
            type: 'subscribe',
            events: events
        });
    }
    
    unsubscribe(events) {
        if (!Array.isArray(events)) {
            events = [events];
        }
        
        events.forEach(event => this.subscriptions.delete(event));
        
        this.send({
            type: 'unsubscribe',
            events: events
        });
    }
    
    ping() {
        this.send({
            type: 'ping',
            timestamp: new Date().toISOString()
        });
    }
    
    requestStatus() {
        this.send({
            type: 'get_status'
        });
    }
    
    on(event, handler) {
        if (!this.eventHandlers.has(event)) {
            this.eventHandlers.set(event, []);
        }
        this.eventHandlers.get(event).push(handler);
    }
    
    off(event, handler) {
        if (this.eventHandlers.has(event)) {
            const handlers = this.eventHandlers.get(event);
            const index = handlers.indexOf(handler);
            if (index > -1) {
                handlers.splice(index, 1);
            }
        }
    }
    
    emit(event, ...args) {
        if (this.eventHandlers.has(event)) {
            this.eventHandlers.get(event).forEach(handler => {
                try {
                    handler(...args);
                } catch (error) {
                    console.error(`Error in event handler for ${event}:`, error);
                }
            });
        }
    }
    
    updateConnectionStatus(connected, message = '') {
        const statusElement = document.getElementById('connection-status');
        if (statusElement) {
            const badge = statusElement.querySelector('.badge');
            if (connected) {
                badge.className = 'badge bg-success';
                badge.innerHTML = '<i class="fas fa-wifi"></i> Connected';
            } else {
                badge.className = 'badge bg-danger';
                badge.innerHTML = '<i class="fas fa-wifi"></i> Disconnected';
                if (message) {
                    badge.innerHTML += ` (${message})`;
                }
            }
        }
    }
    
    showNotification(title, message, type = 'info') {
        const toast = document.getElementById('notification-toast');
        if (toast) {
            const toastHeader = toast.querySelector('.toast-header strong');
            const toastBody = toast.querySelector('.toast-body');
            
            if (toastHeader) toastHeader.textContent = title;
            if (toastBody) toastBody.textContent = message;
            
            // Update toast styling based on type
            toast.className = `toast border-${type}`;
            
            const bsToast = new bootstrap.Toast(toast);
            bsToast.show();
        }
    }
    
    setupEventHandlers() {
        // Setup periodic ping
        setInterval(() => {
            if (this.isConnected) {
                this.ping();
            }
        }, 30000); // Ping every 30 seconds
        
        // Handle page visibility change
        document.addEventListener('visibilitychange', () => {
            if (document.visibilityState === 'visible' && !this.isConnected) {
                this.connect();
            }
        });
        
        // Handle window beforeunload
        window.addEventListener('beforeunload', () => {
            if (this.socket) {
                this.socket.close();
            }
        });
    }
    
    disconnect() {
        if (this.socket) {
            this.socket.close();
        }
    }
    
    getConnectionInfo() {
        return {
            isConnected: this.isConnected,
            reconnectAttempts: this.reconnectAttempts,
            subscriptions: Array.from(this.subscriptions)
        };
    }
}

// Global WebSocket manager instance
let wsManager = null;

// Initialize WebSocket when DOM is loaded
document.addEventListener('DOMContentLoaded', function() {
    wsManager = new WebSocketManager();
    
    // Make it globally available
    window.wsManager = wsManager;
});

// Export for module usage
if (typeof module !== 'undefined' && module.exports) {
    module.exports = WebSocketManager;
}
