"""
Database operations for Smart Parking System
"""

from sqlalchemy import create_engine, func, and_, or_, desc, asc
from sqlalchemy.orm import sessionmaker
from sqlalchemy.exc import SQLAlchemyError
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional
import logging

from .models import (
    Vehicle, ParkingSlot, Camera, DetectionLog, 
    SystemMetrics, ParkingEvent, Base
)
from ..utils.logger import get_logger


class DatabaseManager:
    """Database operations manager"""
    
    def __init__(self, connection_string: str):
        """
        Initialize database manager
        
        Args:
            connection_string: Database connection string
        """
        self.logger = get_logger(self.__class__.__name__)
        self.connection_string = connection_string
        
        try:
            # Create engine
            self.engine = create_engine(connection_string, echo=False)
            
            # Create tables
            Base.metadata.create_all(self.engine)
            
            # Create session factory
            self.SessionLocal = sessionmaker(bind=self.engine)
            
            self.logger.info(f"Database connected: {connection_string}")
            
        except Exception as e:
            self.logger.error(f"Database connection failed: {e}")
            raise
    
    def get_session(self):
        """Get database session"""
        return self.SessionLocal()
    
    # Vehicle operations
    def add_vehicle_entry(
        self, 
        license_plate: str, 
        vehicle_type: str,
        entry_camera: str,
        track_id: int = None,
        confidence: float = None,
        parking_zone: str = None,
        slot_number: int = None,
        metadata: Dict[str, Any] = None
    ) -> Optional[Vehicle]:
        """Add new vehicle entry"""
        session = self.get_session()
        try:
            # Check if vehicle already exists and is active
            existing = session.query(Vehicle).filter(
                and_(
                    Vehicle.license_plate == license_plate,
                    Vehicle.is_active == True
                )
            ).first()
            
            if existing:
                # Vehicle already in parking lot
                self.logger.warning(f"Vehicle {license_plate} already exists")
                return existing
            
            # Create new vehicle entry
            vehicle = Vehicle(
                license_plate=license_plate,
                vehicle_type=vehicle_type,
                entry_time=datetime.utcnow(),
                entry_camera=entry_camera,
                track_id=track_id,
                confidence_score=confidence,
                parking_zone=parking_zone,
                slot_number=slot_number,
                is_active=True
            )
            
            if metadata:
                vehicle.set_metadata(metadata)
            
            session.add(vehicle)
            session.commit()
            
            self.logger.info(f"Vehicle entry added: {license_plate}")
            return vehicle
            
        except SQLAlchemyError as e:
            session.rollback()
            self.logger.error(f"Error adding vehicle entry: {e}")
            return None
        finally:
            session.close()
    
    def add_vehicle_exit(
        self,
        license_plate: str,
        exit_camera: str,
        exit_time: datetime = None
    ) -> Optional[Vehicle]:
        """Add vehicle exit"""
        session = self.get_session()
        try:
            vehicle = session.query(Vehicle).filter(
                and_(
                    Vehicle.license_plate == license_plate,
                    Vehicle.is_active == True
                )
            ).first()
            
            if not vehicle:
                self.logger.warning(f"Vehicle {license_plate} not found for exit")
                return None
            
            # Update vehicle exit info
            vehicle.exit_time = exit_time or datetime.utcnow()
            vehicle.exit_camera = exit_camera
            vehicle.is_active = False
            vehicle.updated_at = datetime.utcnow()
            
            # Free up parking slot if assigned
            if vehicle.slot_number:
                slot = session.query(ParkingSlot).filter(
                    ParkingSlot.slot_number == vehicle.slot_number
                ).first()
                if slot:
                    slot.is_occupied = False
                    slot.occupied_by = None
                    slot.occupied_since = None
                    slot.status = 'available'
            
            session.commit()
            
            self.logger.info(f"Vehicle exit recorded: {license_plate}")
            return vehicle
            
        except SQLAlchemyError as e:
            session.rollback()
            self.logger.error(f"Error recording vehicle exit: {e}")
            return None
        finally:
            session.close()
    
    def get_vehicle_by_plate(self, license_plate: str) -> Optional[Dict[str, Any]]:
        """Get vehicle by license plate"""
        session = self.get_session()
        try:
            vehicle = session.query(Vehicle).filter(
                Vehicle.license_plate == license_plate
            ).order_by(desc(Vehicle.created_at)).first()
            
            return vehicle.to_dict() if vehicle else None
            
        except SQLAlchemyError as e:
            self.logger.error(f"Error getting vehicle: {e}")
            return None
        finally:
            session.close()
    
    def get_all_vehicles(self, active_only: bool = False) -> List[Dict[str, Any]]:
        """Get all vehicles"""
        session = self.get_session()
        try:
            query = session.query(Vehicle)
            
            if active_only:
                query = query.filter(Vehicle.is_active == True)
            
            vehicles = query.order_by(desc(Vehicle.entry_time)).all()
            return [vehicle.to_dict() for vehicle in vehicles]
            
        except SQLAlchemyError as e:
            self.logger.error(f"Error getting vehicles: {e}")
            return []
        finally:
            session.close()
    
    def get_recent_vehicles(self, limit: int = 10) -> List[Dict[str, Any]]:
        """Get recent vehicles"""
        session = self.get_session()
        try:
            vehicles = session.query(Vehicle).order_by(
                desc(Vehicle.created_at)
            ).limit(limit).all()
            
            return [vehicle.to_dict() for vehicle in vehicles]
            
        except SQLAlchemyError as e:
            self.logger.error(f"Error getting recent vehicles: {e}")
            return []
        finally:
            session.close()
    
    def search_vehicles(self, query: str) -> List[Dict[str, Any]]:
        """Search vehicles by license plate"""
        session = self.get_session()
        try:
            vehicles = session.query(Vehicle).filter(
                Vehicle.license_plate.like(f"%{query}%")
            ).order_by(desc(Vehicle.created_at)).limit(20).all()
            
            return [vehicle.to_dict() for vehicle in vehicles]
            
        except SQLAlchemyError as e:
            self.logger.error(f"Error searching vehicles: {e}")
            return []
        finally:
            session.close()
    
    def get_vehicles_paginated(self, page: int, limit: int) -> Dict[str, Any]:
        """Get vehicles with pagination"""
        session = self.get_session()
        try:
            offset = (page - 1) * limit
            
            total = session.query(Vehicle).count()
            vehicles = session.query(Vehicle).order_by(
                desc(Vehicle.created_at)
            ).offset(offset).limit(limit).all()
            
            return {
                "vehicles": [vehicle.to_dict() for vehicle in vehicles],
                "total": total,
                "page": page,
                "limit": limit,
                "total_pages": (total + limit - 1) // limit
            }
            
        except SQLAlchemyError as e:
            self.logger.error(f"Error getting paginated vehicles: {e}")
            return {"vehicles": [], "total": 0, "page": page, "limit": limit, "total_pages": 0}
        finally:
            session.close()
    
    # Parking slot operations
    def initialize_parking_slots(self, total_slots: int, zones: List[Dict[str, Any]]):
        """Initialize parking slots"""
        session = self.get_session()
        try:
            # Clear existing slots
            session.query(ParkingSlot).delete()
            
            slot_number = 1
            for zone in zones:
                zone_name = zone["name"]
                zone_slots = zone["slots"]
                
                for i in range(zone_slots):
                    slot = ParkingSlot(
                        slot_number=slot_number,
                        zone_name=zone_name,
                        is_occupied=False,
                        status='available'
                    )
                    
                    if "coordinates" in zone:
                        slot.set_coordinates(zone["coordinates"])
                    
                    session.add(slot)
                    slot_number += 1
            
            session.commit()
            self.logger.info(f"Initialized {slot_number-1} parking slots")
            
        except SQLAlchemyError as e:
            session.rollback()
            self.logger.error(f"Error initializing parking slots: {e}")
        finally:
            session.close()
    
    def occupy_slot(self, slot_number: int, license_plate: str) -> bool:
        """Occupy a parking slot"""
        session = self.get_session()
        try:
            slot = session.query(ParkingSlot).filter(
                ParkingSlot.slot_number == slot_number
            ).first()
            
            if not slot or slot.is_occupied:
                return False
            
            slot.is_occupied = True
            slot.occupied_by = license_plate
            slot.occupied_since = datetime.utcnow()
            slot.status = 'occupied'
            
            session.commit()
            return True
            
        except SQLAlchemyError as e:
            session.rollback()
            self.logger.error(f"Error occupying slot: {e}")
            return False
        finally:
            session.close()
    
    def free_slot(self, slot_number: int) -> bool:
        """Free a parking slot"""
        session = self.get_session()
        try:
            slot = session.query(ParkingSlot).filter(
                ParkingSlot.slot_number == slot_number
            ).first()
            
            if not slot:
                return False
            
            slot.is_occupied = False
            slot.occupied_by = None
            slot.occupied_since = None
            slot.status = 'available'
            
            session.commit()
            return True
            
        except SQLAlchemyError as e:
            session.rollback()
            self.logger.error(f"Error freeing slot: {e}")
            return False
        finally:
            session.close()
    
    def get_parking_status(self) -> Dict[str, Any]:
        """Get current parking status"""
        session = self.get_session()
        try:
            total_slots = session.query(ParkingSlot).count()
            occupied_slots = session.query(ParkingSlot).filter(
                ParkingSlot.is_occupied == True
            ).count()
            available_slots = total_slots - occupied_slots
            
            # Get zone statistics
            zones = session.query(
                ParkingSlot.zone_name,
                func.count(ParkingSlot.id).label('total'),
                func.sum(func.cast(ParkingSlot.is_occupied, int)).label('occupied')
            ).group_by(ParkingSlot.zone_name).all()
            
            zone_stats = []
            for zone in zones:
                zone_stats.append({
                    "name": zone.zone_name,
                    "total": zone.total,
                    "occupied": zone.occupied or 0,
                    "available": zone.total - (zone.occupied or 0),
                    "occupancy_rate": ((zone.occupied or 0) / zone.total * 100) if zone.total > 0 else 0
                })
            
            # Get active vehicles count
            active_vehicles = session.query(Vehicle).filter(
                Vehicle.is_active == True
            ).count()
            
            return {
                "total_slots": total_slots,
                "occupied_slots": occupied_slots,
                "available_slots": available_slots,
                "occupancy_rate": (occupied_slots / total_slots * 100) if total_slots > 0 else 0,
                "active_vehicles": active_vehicles,
                "zones": zone_stats,
                "last_updated": datetime.utcnow().isoformat()
            }
            
        except SQLAlchemyError as e:
            self.logger.error(f"Error getting parking status: {e}")
            return {
                "total_slots": 0,
                "occupied_slots": 0,
                "available_slots": 0,
                "occupancy_rate": 0,
                "active_vehicles": 0,
                "zones": [],
                "last_updated": datetime.utcnow().isoformat()
            }
        finally:
            session.close()
    
    def get_zones_status(self) -> List[Dict[str, Any]]:
        """Get status of all parking zones"""
        session = self.get_session()
        try:
            zones = session.query(
                ParkingSlot.zone_name,
                func.count(ParkingSlot.id).label('total'),
                func.sum(func.cast(ParkingSlot.is_occupied, int)).label('occupied')
            ).group_by(ParkingSlot.zone_name).all()
            
            zone_stats = []
            for zone in zones:
                # Get slots details for this zone
                slots = session.query(ParkingSlot).filter(
                    ParkingSlot.zone_name == zone.zone_name
                ).all()
                
                zone_stats.append({
                    "name": zone.zone_name,
                    "total": zone.total,
                    "occupied": zone.occupied or 0,
                    "available": zone.total - (zone.occupied or 0),
                    "occupancy_rate": ((zone.occupied or 0) / zone.total * 100) if zone.total > 0 else 0,
                    "slots": [slot.to_dict() for slot in slots]
                })
            
            return zone_stats
            
        except SQLAlchemyError as e:
            self.logger.error(f"Error getting zones status: {e}")
            return []
        finally:
            session.close()
    
    # Statistics operations
    def get_daily_statistics(self, days: int = 7) -> List[Dict[str, Any]]:
        """Get daily statistics"""
        session = self.get_session()
        try:
            end_date = datetime.utcnow().date()
            start_date = end_date - timedelta(days=days-1)
            
            stats = []
            current_date = start_date
            
            while current_date <= end_date:
                # Count entries and exits for this date
                day_start = datetime.combine(current_date, datetime.min.time())
                day_end = datetime.combine(current_date, datetime.max.time())
                
                entries = session.query(Vehicle).filter(
                    and_(
                        Vehicle.entry_time >= day_start,
                        Vehicle.entry_time <= day_end
                    )
                ).count()
                
                exits = session.query(Vehicle).filter(
                    and_(
                        Vehicle.exit_time >= day_start,
                        Vehicle.exit_time <= day_end
                    )
                ).count()
                
                # Calculate average parking duration
                vehicles_with_exit = session.query(Vehicle).filter(
                    and_(
                        Vehicle.entry_time >= day_start,
                        Vehicle.entry_time <= day_end,
                        Vehicle.exit_time.isnot(None)
                    )
                ).all()
                
                avg_duration = 0
                if vehicles_with_exit:
                    total_duration = sum(
                        (v.exit_time - v.entry_time).total_seconds() / 60 
                        for v in vehicles_with_exit
                    )
                    avg_duration = total_duration / len(vehicles_with_exit)
                
                stats.append({
                    "date": current_date.isoformat(),
                    "entries": entries,
                    "exits": exits,
                    "net_change": entries - exits,
                    "avg_duration_minutes": round(avg_duration, 2)
                })
                
                current_date += timedelta(days=1)
            
            return stats
            
        except SQLAlchemyError as e:
            self.logger.error(f"Error getting daily statistics: {e}")
            return []
        finally:
            session.close()
    
    def get_hourly_statistics(self, date: str = None) -> List[Dict[str, Any]]:
        """Get hourly statistics for a specific date"""
        session = self.get_session()
        try:
            if date:
                target_date = datetime.strptime(date, '%Y-%m-%d').date()
            else:
                target_date = datetime.utcnow().date()
            
            day_start = datetime.combine(target_date, datetime.min.time())
            day_end = datetime.combine(target_date, datetime.max.time())
            
            stats = []
            
            for hour in range(24):
                hour_start = day_start + timedelta(hours=hour)
                hour_end = hour_start + timedelta(hours=1)
                
                entries = session.query(Vehicle).filter(
                    and_(
                        Vehicle.entry_time >= hour_start,
                        Vehicle.entry_time < hour_end
                    )
                ).count()
                
                exits = session.query(Vehicle).filter(
                    and_(
                        Vehicle.exit_time >= hour_start,
                        Vehicle.exit_time < hour_end
                    )
                ).count()
                
                stats.append({
                    "hour": hour,
                    "entries": entries,
                    "exits": exits,
                    "net_change": entries - exits
                })
            
            return stats
            
        except SQLAlchemyError as e:
            self.logger.error(f"Error getting hourly statistics: {e}")
            return []
        finally:
            session.close()
    
    def get_occupancy_trends(self, days: int = 7) -> List[Dict[str, Any]]:
        """Get occupancy trends over time"""
        session = self.get_session()
        try:
            end_date = datetime.utcnow()
            start_date = end_date - timedelta(days=days)
            
            # This is a simplified version - in practice you'd want to sample at regular intervals
            # and calculate occupancy at each point
            trends = []
            current_time = start_date
            
            while current_time <= end_date:
                # Count vehicles that were in parking lot at this time
                occupied = session.query(Vehicle).filter(
                    and_(
                        Vehicle.entry_time <= current_time,
                        or_(
                            Vehicle.exit_time.is_(None),
                            Vehicle.exit_time > current_time
                        )
                    )
                ).count()
                
                trends.append({
                    "timestamp": current_time.isoformat(),
                    "occupied_slots": occupied,
                    "occupancy_rate": (occupied / 300 * 100)  # Assuming 300 total slots
                })
                
                current_time += timedelta(hours=1)  # Sample every hour
            
            return trends[-168:] if len(trends) > 168 else trends  # Last 7 days * 24 hours
            
        except SQLAlchemyError as e:
            self.logger.error(f"Error getting occupancy trends: {e}")
            return []
        finally:
            session.close()
    
    # System metrics operations
    def log_metric(
        self,
        metric_name: str,
        metric_value: float,
        metric_unit: str = None,
        camera_id: str = None
    ):
        """Log system metric"""
        session = self.get_session()
        try:
            metric = SystemMetrics(
                metric_name=metric_name,
                metric_value=metric_value,
                metric_unit=metric_unit,
                camera_id=camera_id,
                timestamp=datetime.utcnow()
            )
            
            session.add(metric)
            session.commit()
            
        except SQLAlchemyError as e:
            session.rollback()
            self.logger.error(f"Error logging metric: {e}")
        finally:
            session.close()
    
    def log_detection(
        self,
        camera_id: str,
        detection_type: str,
        confidence: float,
        bbox: List[int],
        class_name: str = None,
        license_plate: str = None,
        track_id: int = None,
        processing_time_ms: float = None
    ):
        """Log detection result"""
        session = self.get_session()
        try:
            detection = DetectionLog(
                camera_id=camera_id,
                detection_type=detection_type,
                confidence=confidence,
                bbox_x1=bbox[0],
                bbox_y1=bbox[1],
                bbox_x2=bbox[2],
                bbox_y2=bbox[3],
                class_name=class_name,
                license_plate=license_plate,
                track_id=track_id,
                processing_time_ms=processing_time_ms,
                frame_timestamp=datetime.utcnow()
            )
            
            session.add(detection)
            session.commit()
            
        except SQLAlchemyError as e:
            session.rollback()
            self.logger.error(f"Error logging detection: {e}")
        finally:
            session.close()
    
    def cleanup_old_data(self, days_to_keep: int = 30):
        """Clean up old data"""
        session = self.get_session()
        try:
            cutoff_date = datetime.utcnow() - timedelta(days=days_to_keep)
            
            # Clean up old detection logs
            deleted_detections = session.query(DetectionLog).filter(
                DetectionLog.frame_timestamp < cutoff_date
            ).delete()
            
            # Clean up old metrics
            deleted_metrics = session.query(SystemMetrics).filter(
                SystemMetrics.timestamp < cutoff_date
            ).delete()
            
            session.commit()
            
            self.logger.info(f"Cleaned up {deleted_detections} detection logs and {deleted_metrics} metrics")
            
        except SQLAlchemyError as e:
            session.rollback()
            self.logger.error(f"Error cleaning up old data: {e}")
        finally:
            session.close()
