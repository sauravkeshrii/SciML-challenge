#!/usr/bin/env python3
"""
🍎 GEN-SHM REVOLUTIONARY DEMO - TRUE STEVE JOBS STYLE
This creates an instantly understandable "before and after" visualization
that shows WHY people need Gen-SHM protection for their drones
"""

import sys
from pathlib import Path
import os
import time
import threading
import math

# Add src to path properly
project_root = Path(__file__).parent.parent
src_path = project_root / 'src'
sys.path.insert(0, str(src_path))
os.chdir(str(project_root))

try:
    import tkinter as tk
    from tkinter import ttk
    import pygame
except ImportError:
    import subprocess
    subprocess.check_call([sys.executable, "-m", "pip", "install", "pygame"])
    import pygame
    import tkinter as tk
    from tkinter import ttk

class RevolutionaryDroneDemo:
    def __init__(self, root):
        self.root = root
        self.root.title("Gen-SHM Mission Critical")
        self.root.geometry("1200x800")
        self.root.configure(bg='#000000')
        
        # Mission state
        self.mission_active = False
        self.mission_start_time = 0
        
        # Create mission-focused design
        self.create_revolutionary_design()
        
    def create_revolutionary_design(self):
        """Create minimalist Apple-style interface with mission context"""
        # Pure black background - maximum contrast
        main_frame = tk.Frame(self.root, bg='#000000')
        main_frame.pack(fill=tk.BOTH, expand=True)
        
        # Mission title
        self.title_label = tk.Label(
            main_frame,
            text="CRITICAL RESCUE MISSION",
            font=("Helvetica", 36, "bold"),
            fg="#ffffff",
            bg="#000000"
        )
        self.title_label.pack(pady=(30, 5))
        
        # Mission context
        self.subtitle_label = tk.Label(
            main_frame,
            text="Medical supplies to remote disaster zone",
            font=("Helvetica", 16),
            fg="#aaaaaa",
            bg="#000000"
        )
        self.subtitle_label.pack(pady=(0, 5))
        
        # Critical mission timer/status
        self.mission_status = tk.Label(
            main_frame,
            text="Mission Elapsed: 00:00:00",
            font=("Helvetica", 14),
            fg="#ff9900",
            bg="#000000"
        )
        self.mission_status.pack(pady=(0, 20))
        
        # Clean split view
        viz_frame = tk.Frame(main_frame, bg='#000000')
        viz_frame.pack(fill=tk.BOTH, expand=True)
        
        # LEFT: Mission failure
        self.before_frame = tk.Frame(viz_frame, bg='#000000')
        self.before_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        
        self.before_title = tk.Label(
            self.before_frame,
            text="WITHOUT Gen-SHM",
            font=("Helvetica", 16, "bold"),
            fg="#ff3333",
            bg="#000000"
        )
        self.before_title.pack(pady=5)
        
        self.before_canvas = tk.Canvas(self.before_frame, bg='#000000', highlightthickness=0)
        self.before_canvas.pack(fill=tk.BOTH, expand=True)
        
        # RIGHT: Mission success
        self.after_frame = tk.Frame(viz_frame, bg='#000000')
        self.after_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)
        
        self.after_title = tk.Label(
            self.after_frame,
            text="WITH Gen-SHM",
            font=("Helvetica", 16, "bold"),
            fg="#00ff00",
            bg="#000000"
        )
        self.after_title.pack(pady=5)
        
        self.after_canvas = tk.Canvas(self.after_frame, bg='#000000', highlightthickness=0)
        self.after_canvas.pack(fill=tk.BOTH, expand=True)
        
        # Mission critical status
        self.status_label = tk.Label(
            main_frame,
            text="Mission Status: Proceeding Normally",
            font=("Helvetica", 14, "bold"),
            fg="#ffffff",
            bg="#000000"
        )
        self.status_label.pack(pady=15)
        
        # Single elegant button
        self.action_button = tk.Button(
            main_frame,
            text="Launch Mission",
            font=("Helvetica", 16, "bold"),
            fg="#ffffff",
            bg="#333333",
            activebackground="#555555",
            activeforeground="#ffffff",
            relief="flat",
            padx=40,
            pady=15,
            command=self.toggle_mission
        )
        self.action_button.pack()
        
        # Initialize state
        self.draw_initial_state()
        
    def draw_initial_state(self):
        """Draw initial simple state"""
        self.before_canvas.delete("all")
        self.after_canvas.delete("all")
        
        width_before = self.before_canvas.winfo_width()
        width_after = self.after_canvas.winfo_width()
        height = self.before_canvas.winfo_height()
        
        if width_before < 10 or height < 10:
            self.root.after(100, self.draw_initial_state)
            return
        
        # Simple ground line
        ground_y = height - 50
        self.before_canvas.create_line(0, ground_y, width_before, ground_y, fill="#333333")
        self.after_canvas.create_line(0, ground_y, width_after, ground_y, fill="#333333")
        
        # Simple identical drones
        drone_x = width_before // 2
        drone_y = ground_y - 100
        
        self.draw_drone(self.before_canvas, drone_x, drone_y, "#ff9900")
        self.draw_drone(self.after_canvas, drone_x, drone_y, "#00ff00")
        
    def draw_drone(self, canvas, x, y, color):
        """Draw minimalist drone representation"""
        size = 20
        
        # Simple drone body
        canvas.create_polygon(
            x-size, y-size//3,
            x+size, y-size//3,
            x+size//2, y+size//2,
            x-size//2, y+size//2,
            fill=color, outline="#ffffff", width=2
        )
        
        # Simple propellers
        prop_radius = 5
        positions = [(-size, -size//3), (size, -size//3), (-size//2, size//2), (size//2, size//2)]
        
        for px, py in positions:
            canvas.create_oval(x+px-prop_radius, y+py-prop_radius,
                              x+px+prop_radius, y+py+prop_radius,
                              fill="#ffffff", outline=color, width=1)
    
    def toggle_mission(self):
        """Toggle the mission demonstration"""
        if self.mission_active:
            self.stop_mission()
        else:
            self.start_mission()
    
    def start_mission(self):
        """Start mission demonstration"""
        self.mission_active = True
        self.mission_start_time = time.time()
        self.action_button.config(text="Abort Mission", bg="#ff3333")
        self.status_label.config(text="Mission in Progress - Lives Depend on Success")
        self.title_label.config(text="CRITICAL RESCUE MISSION IN PROGRESS")
        
        # Start mission animation
        self.mission_thread = threading.Thread(target=self.mission_animation)
        self.mission_thread.daemon = True
        self.mission_thread.start()
        
        # Start mission timer
        self.update_mission_timer()
    
    def stop_mission(self):
        """Stop mission demonstration"""
        self.mission_active = False
        self.action_button.config(text="Launch Mission", bg="#333333")
        self.status_label.config(text="Mission Status: Standby")
        self.title_label.config(text="CRITICAL RESCUE MISSION")
        self.mission_status.config(text="Mission Elapsed: 00:00:00")
        self.draw_initial_state()
    
    def update_mission_timer(self):
        """Update mission elapsed time display"""
        if self.mission_active:
            elapsed = int(time.time() - self.mission_start_time)
            hours = elapsed // 3600
            minutes = (elapsed % 3600) // 60
            seconds = elapsed % 60
            time_str = f"Mission Elapsed: {hours:02d}:{minutes:02d}:{seconds:02d}"
            self.mission_status.config(text=time_str)
            self.root.after(1000, self.update_mission_timer)
    
    def draw_initial_state(self):
        """Draw initial simple state"""
        self.before_canvas.delete("all")
        self.after_canvas.delete("all")
        
        width_before = self.before_canvas.winfo_width()
        width_after = self.after_canvas.winfo_width()
        height = self.before_canvas.winfo_height()
        
        if width_before < 10 or height < 10:
            self.root.after(100, self.draw_initial_state)
            return
        
        # Simple ground line
        ground_y = height - 50
        self.before_canvas.create_line(0, ground_y, width_before, ground_y, fill="#333333")
        self.after_canvas.create_line(0, ground_y, width_after, ground_y, fill="#333333")
        
        # Simple identical drones
        drone_x = width_before // 2
        drone_y = ground_y - 100
        
        self.draw_drone(self.before_canvas, drone_x, drone_y, "#ff9900")
        self.draw_drone(self.after_canvas, drone_x, drone_y, "#00ff00")
        
    def mission_animation(self):
        """Mission-focused animation showing critical difference"""
        try:
            width_before = self.before_canvas.winfo_width()
            width_after = self.after_canvas.winfo_width()
            height = self.before_canvas.winfo_height()
            ground_y = height - 50
            
            start_time = time.time()
            
            while self.mission_active:
                elapsed = time.time() - start_time
                
                # Clear canvases
                self.before_canvas.delete("all")
                self.after_canvas.delete("all")
                
                # Simple ground
                self.before_canvas.create_line(0, ground_y, width_before, ground_y, fill="#333333")
                self.after_canvas.create_line(0, ground_y, width_after, ground_y, fill="#333333")
                
                # Mission payload visualization
                # Medical cross symbol to show rescue mission
                self.before_canvas.create_text(width_before//2, ground_y - 150, 
                                             text="⚕", font=("Arial", 20), fill="#ff9900")
                self.after_canvas.create_text(width_after//2, ground_y - 150,
                                            text="⚕", font=("Arial", 20), fill="#00ff00")
                
                # Protected drone: mission success
                protected_x = width_after // 2 + math.sin(elapsed * 0.4) * 12
                protected_y = ground_y - 120 + math.sin(elapsed * 0.2) * 4
                self.draw_drone(self.after_canvas, protected_x, protected_y, "#00ff00")
                
                # Unprotected drone: mission failure scenario
                mission_critical_time = min(elapsed / 15.0, 1.0)  # Mission critical phase
                
                if mission_critical_time < 0.6:
                    # Early mission - normal flight
                    unsafe_x = width_before // 2 + math.sin(elapsed * 0.5) * 12
                    unsafe_y = ground_y - 120 + math.sin(elapsed * 0.3) * 4
                    mission_status = "Medical delivery proceeding normally..."
                elif mission_critical_time < 0.8:
                    # Developing issues
                    instability = (mission_critical_time - 0.6) / 0.2
                    unsafe_x = width_before // 2 + math.sin(elapsed * (0.5 + instability)) * (12 + instability * 20)
                    unsafe_y = ground_y - 120 + math.sin(elapsed * (0.3 + instability * 0.5)) * (4 + instability * 10)
                    mission_status = "WARNING: Flight instability detected..."
                else:
                    # Critical failure
                    failure_progress = (mission_critical_time - 0.8) / 0.2
                    unsafe_x = width_before // 2 + math.sin(elapsed * 2) * (25 + failure_progress * 30) + (random.random() - 0.5) * 20 * failure_progress
                    unsafe_y = ground_y - 120 + math.sin(elapsed) * (15 + failure_progress * 20) - elapsed * 10 * failure_progress
                    
                    if failure_progress > 0.7:
                        mission_status = "💥 CRITICAL FAILURE: Medical cargo LOST!"
                    else:
                        mission_status = "EMERGENCY: Control systems failing..."
                
                self.draw_drone(self.before_canvas, unsafe_x, unsafe_y, "#ff9900")
                
                # Update mission status
                self.root.after(0, lambda s=mission_status: self.status_label.config(text=s, fg="#ff3333" if "CRITICAL" in s or "WARNING" in s else "#ffffff"))
                
                time.sleep(1/30)  # 30 FPS
                
        except Exception as e:
            print(f"Mission animation error: {e}")
            self.mission_active = False
    
    def animate_extended_phase(self, phase, elapsed, width_before, width_after, ground_y, danger_level, protection_level, altitude_factor):
        """Animate extended phase with complete flight profile and dramatic crash"""
        # Protected drone (after) - complete mission profile
        if phase <= 3:
            # Launch and climb phase
            climb_progress = min(elapsed / 10.0, 1.0)  # First 10 seconds climbing
            protected_altitude = ground_y - 150 - (120 * climb_progress * altitude_factor)
            protected_x = width_after // 2 + math.sin(elapsed * 0.5) * 8
            protected_y = protected_altitude + math.sin(elapsed * 0.3) * 4
        elif phase <= 10:
            # Mission phase - cruising at operational altitude
            mission_progress = (elapsed + (phase - 4) * 4.5) / 30.0  # 30 seconds total mission time
            protected_x = width_after // 2 + math.sin(mission_progress * 2) * 25
            protected_y = ground_y - 250 + math.sin(elapsed * 0.2) * 6
        elif phase <= 12:
            # Descent initiation
            descent_progress = min(elapsed / 8.0, 1.0)
            protected_x = width_after // 2 + math.sin(elapsed * 0.4) * 12
            protected_y = ground_y - 250 + (100 * descent_progress) + math.sin(elapsed * 0.25) * 4
        elif phase <= 14:
            # Approach and landing
            landing_progress = min(elapsed / 6.0, 1.0)
            protected_x = width_after // 2 + math.sin(elapsed * 0.2) * 5
            protected_y = ground_y - 150 + (150 * landing_progress)
        else:
            # Landed
            protected_x = width_after // 2
            protected_y = ground_y - 20
        
        # Unprotected drone (before) - progressive failure leading to spectacular crash
        if phase <= 2:
            # Phase 0-2: Normal launch and climb
            climb_progress = min(elapsed / 8.0, 1.0)
            unsafe_altitude = ground_y - 150 - (120 * climb_progress * 0.8)  # Slightly slower climb
            unsafe_x = width_before // 2 + math.sin(elapsed * 0.6) * 10
            unsafe_y = unsafe_altitude + math.sin(elapsed * 0.4) * 5
        elif phase <= 5:
            # Phase 3-5: Beginning instability
            instability = (phase - 2) / 4.0
            unsafe_x = width_before // 2 + math.sin(elapsed * (0.6 + instability)) * (10 + instability * 20)
            unsafe_y = ground_y - 220 + math.sin(elapsed * (0.4 + instability * 0.5)) * (5 + instability * 15)
        elif phase <= 8:
            # Phase 6-8: Severe失控and erratic movement
            chaos_progress = (phase - 5) / 4.0
            unsafe_x = width_before // 2 + math.sin(elapsed * 2) * (20 + chaos_progress * 40) + (random.random() - 0.5) * 30 * chaos_progress
            unsafe_y = ground_y - 200 + math.sin(elapsed * 1.5) * (15 + chaos_progress * 30) - elapsed * 15 * chaos_progress
        elif phase <= 9:
            # Phase 9: Pre-crash失控- wild spinning and dropping
            pre_crash_time = min(elapsed / 1.5, 1.0)
            spin_speed = 5 + pre_crash_time * 10
            unsafe_x = width_before // 2 + math.sin(elapsed * spin_speed) * (50 + pre_crash_time * 100) + (random.random() - 0.5) * 60
            unsafe_y = ground_y - 180 + math.sin(elapsed * spin_speed * 0.5) * 50 - elapsed * 40 * pre_crash_time
        elif phase == 10:
            # Phase 10: Spectacular crash sequence
            crash_progress = min(elapsed / 2.0, 1.0)
            # Explosive outward motion
            explosion_force = math.sin(crash_progress * math.pi) * 200
            unsafe_x = width_before // 2 + (random.random() - 0.5) * explosion_force
            unsafe_y = ground_y - 150 - (crash_progress * 400) + abs(random.random() - 0.5) * 100
        else:
            # Phase 11+: Post-crash debris field
            unsafe_x = width_before // 2 + (random.random() - 0.5) * 150
            unsafe_y = ground_y - 15  # Scattered on ground
            
        # Update canvases with enhanced effects
        self.root.after(0, lambda: self.update_extended_canvases(
            protected_x, protected_y, unsafe_x, unsafe_y, phase, elapsed, 
            danger_level, protection_level, altitude_factor
        ))
        
        # Enhanced dramatic insight text with mission progression
        insight_messages = [
            "MISSION LAUNCH: Both drones achieve perfect liftoff...",
            "CLIMBING TO ALTITUDE: Operational ceiling reached at 100 meters...",
            "CRUISING PHASE: Micro-fractures invisible to traditional inspection...",
            "DAMAGE PROPAGATION: Conventional systems remain oblivious...",
            "GEN-SHM DETECTION: Neural network identifies anomalies in real-time...",
            "COMPENSATION ENGAGED: Adaptive algorithms counteract structural changes...",
            "UNPROTECTED DRONE: Instability growing exponentially...",
            "CONTROL SYSTEMS FAILING: Gyroscopes and accelerometers compromised...",
            "CRASH IMMINENT: All emergency protocols exhausted...",
            "💥 CATASTROPHIC FAILURE: Drone disintegrates in spectacular explosion!",
            "DEBRIS FIELD: $50,000 investment scattered across 200 square meters...",
            "✅ MISSION SUCCESS: Gen-SHM protected drone continues operations flawlessly...",
            "DESCENT INITIATED: Automated landing sequence commencing...",
            "APPROACH PHASE: Precise navigation to designated landing zone...",
            "SAFE TOUCHDOWN: Mission completed. Asset preserved. Lives protected."
        ]
        
        if int(elapsed * 1.5) % 2 == 0 and phase < len(insight_messages):
            self.root.after(0, lambda: self.insight_text.config(
                text=insight_messages[phase]
            ))
        
        # Mission status title evolution
        title_messages = [
            "MISSION START: Perfect Launch Conditions",
            "ASCENT: Reaching Operational Altitude",
            "CRUISE: The Hidden Threat Emerges",
            "THREAT LEVEL INCREASING: Damage Accumulating",
            "GEN-SHM RESPONSE: Intelligent Adaptation Activated",
            "SYSTEM STRESS: Traditional Controls Failing",
            "LOSING CONTROL: Chaos Approaching",
            "EMERGENCY: All Systems Compromised",
            "FINAL MOMENTS: Inevitable Collapse",
            "💥 DEVASTATING EXPLOSION: Mission Terminated",
            "AFTERMATH: Complete System Failure",
            "✅ GEN-SHM VICTORY: Mission Accomplished",
            "RETURN TO BASE: Descent Protocol Engaged",
            "LANDING SEQUENCE: Precision Approach",
            "MISSION COMPLETE: Total Success"
        ]
        
        if int(elapsed) % 2 == 0 and phase < len(title_messages):
            self.root.after(0, lambda: self.title_label.config(
                text=title_messages[phase]
            ))
    
    def update_extended_canvases(self, protected_x, protected_y, unsafe_x, unsafe_y, phase, elapsed, danger_level, protection_level, altitude_factor):
        """Update canvases with extended flight profile and spectacular crash effects"""
        try:
            # Clear and redraw
            self.before_canvas.delete("all")
            self.after_canvas.delete("all")
                
            width_before = self.before_canvas.winfo_width()
            width_after = self.after_canvas.winfo_width()
            height = self.before_canvas.winfo_height()
            ground_y = height - 70
                
            # Dynamic atmospheric backgrounds
            # Before: Progressive storm intensification
            if phase <= 5:
                storm_intensity = int(20 + 20 * danger_level)
                storm_color = f"#{storm_intensity:02x}{storm_intensity//2:02x}{storm_intensity*2:02x}"
            elif phase <= 9:
                # Pre-crash lightning storm
                flash_intensity = int(40 + 60 * math.sin(elapsed * 20))
                storm_color = f"#{flash_intensity:02x}{flash_intensity//3:02x}ff"
            else:
                # Post-crash aftermath
                storm_color = "#110022"
                
            self.before_canvas.create_rectangle(0, 0, width_before, height, fill=storm_color, outline="")
                
            # After: Progressive sky improvement
            if phase <= 10:
                sky_blue = int(34 + 34 * (1 - protection_level * 0.5))
                sky_green = int(34 * (1 - protection_level * 0.3))
                sky_color = f"#{sky_green:02x}{sky_blue:02x}ff"
            else:
                # Landing phase - golden sunset
                sunset_gold = int(68 + 34 * math.sin(elapsed))
                sky_color = f"#ff{sunset_gold:02x}00"
                
            self.after_canvas.create_rectangle(0, 0, width_after, height, fill=sky_color, outline="")
                
            # Enhanced ground with mission-specific effects
            # Before: Battle damage terrain
            self.before_canvas.create_rectangle(0, ground_y, width_before, height, fill="#331111", outline="")
                
            # Add progressively worse battlefield effects
            if phase >= 3:
                # Early damage indicators
                crack_count = int(5 + 15 * danger_level)
                for i in range(crack_count):
                    crack_x = random.randint(0, width_before)
                    crack_length = random.randint(10, 40)
                    self.before_canvas.create_line(crack_x, ground_y, crack_x, ground_y + crack_length,
                                                 fill="#ff3300", width=random.randint(1, 3))
                
            if phase >= 10:
                # Post-crash crater and debris field
                crater_x = width_before // 2
                crater_radius = 40 + int(20 * math.sin(elapsed * 5))
                self.before_canvas.create_oval(crater_x-crater_radius, ground_y-10,
                                             crater_x+crater_radius, ground_y+30,
                                             fill="#000000", outline="#ff6600", width=3)
                    
                # Debris scatter
                for i in range(20):
                    debris_x = crater_x + (random.random() - 0.5) * 150
                    debris_y = ground_y + random.randint(-10, 20)
                    debris_size = random.randint(2, 8)
                    self.before_canvas.create_rectangle(
                        debris_x-debris_size//2, debris_y-debris_size//2,
                        debris_x+debris_size//2, debris_y+debris_size//2,
                        fill="#ff6600", outline="white"
                    )
                
            # After: Mission success landing zone
            self.after_canvas.create_rectangle(0, ground_y, width_after, height, fill="#228B22", outline="")
                
            # Add landing markers and success indicators
            if phase >= 12:
                # Landing zone markers
                marker_positions = [(width_after//2 - 60, ground_y-25), (width_after//2 + 60, ground_y-25)]
                for mx, my in marker_positions:
                    self.after_canvas.create_oval(mx-8, my-8, mx+8, my+8, fill="#ffff00", outline="white", width=2)
                    self.after_canvas.create_text(mx, my, text="LAND", fill="black", font=("Helvetica", 8, "bold"))
                
            if phase >= 14:
                # Mission complete celebration effects
                confetti_count = int(10 + 10 * math.sin(elapsed * 10))
                for i in range(confetti_count):
                    confetti_x = random.randint(0, width_after)
                    confetti_y = random.randint(0, ground_y - 50)
                    colors = ["#ff0000", "#00ff00", "#0000ff", "#ffff00", "#ff00ff"]
                    self.after_canvas.create_oval(confetti_x-2, confetti_y-2, confetti_x+2, confetti_y+2,
                                                fill=random.choice(colors), outline="white")
                
            # Enhanced scenario labels
            if phase <= 9:
                before_label = "MISSION FAILURE: Traditional Drone Systems"
                after_label = "MISSION SUCCESS: Gen-SHM Protected Operations"
            elif phase <= 10:
                before_label = "💥 CATASTROPHIC SYSTEM FAILURE"
                after_label = "✅ CONTINUED MISSION SUCCESS"
            else:
                before_label = "FAILED MISSION: TOTAL LOSS"
                after_label = "MISSION ACCOMPLISHED: SAFE RETURN"
                
            self.before_canvas.create_text(width_before//2, 30, text=before_label,
                                         fill="#ff3333", font=("Helvetica", 16, "bold"))
            self.after_canvas.create_text(width_after//2, 30, text=after_label,
                                        fill="#33ff33", font=("Helvetica", 16, "bold"))
                
            # Draw drones with mission-appropriate damage visualization
            self.draw_drone(self.after_canvas, protected_x, protected_y, "#00ff00", "Protected Mission Drone", protection_level)
            self.draw_drone(self.before_canvas, unsafe_x, unsafe_y, "#ff9900", "Failed Traditional Drone", danger_level)
                
            # Add spectacular crash effects for phase 10
            if phase == 10:
                crash_center_x = width_before // 2
                crash_center_y = ground_y - 100
                    
                # Multi-stage explosion sequence
                explosion_stage = min(elapsed / 2.0, 1.0)
                    
                # Primary explosion
                primary_radius = int(30 + 70 * math.sin(explosion_stage * math.pi))
                self.before_canvas.create_oval(
                    crash_center_x-primary_radius, crash_center_y-primary_radius,
                    crash_center_x+primary_radius, crash_center_y+primary_radius,
                    fill="#ff9900", outline="#ffffff", width=3, stipple="gray50"
                )
                    
                # Secondary fireballs
                if explosion_stage > 0.3:
                    for i in range(5):
                        angle = (i * 72 + elapsed * 50) * math.pi / 180
                        distance = 40 + 60 * explosion_stage
                        fireball_x = crash_center_x + math.cos(angle) * distance
                        fireball_y = crash_center_y + math.sin(angle) * distance
                        fireball_radius = int(15 + 25 * math.sin(explosion_stage * math.pi + i))
                        self.before_canvas.create_oval(
                            fireball_x-fireball_radius, fireball_y-fireball_radius,
                            fireball_x+fireball_radius, fireball_y+fireball_radius,
                            fill="#ff3300", outline="#ffff00", stipple="gray25"
                        )
                    
                # Shockwave effect
                if explosion_stage > 0.5:
                    shockwave_radius = int(80 * explosion_stage)
                    shockwave_intensity = int(255 * (1 - explosion_stage))
                    shockwave_color = f"#{shockwave_intensity:02x}{shockwave_intensity//2:02x}ff"
                    self.before_canvas.create_oval(
                        crash_center_x-shockwave_radius, crash_center_y-shockwave_radius,
                        crash_center_x+shockwave_radius, crash_center_y+shockwave_radius,
                        outline=shockwave_color, width=5
                    )
                    
                # Smoke trails
                if explosion_stage > 0.7:
                    smoke_particles = int(30 * explosion_stage)
                    for i in range(smoke_particles):
                        smoke_x = crash_center_x + (random.random() - 0.5) * 100
                        smoke_y = crash_center_y - elapsed * 30 + (random.random() - 0.5) * 50
                        smoke_size = random.randint(3, 12)
                        self.before_canvas.create_oval(
                            smoke_x-smoke_size, smoke_y-smoke_size,
                            smoke_x+smoke_size, smoke_y+smoke_size,
                            fill="#333333", outline="white", stipple="gray12"
                        )
                
            # Enhanced telemetry displays
            if phase >= 2:
                # Before drone telemetry (degrading)
                if phase <= 9:
                    alt_before = max(0, 100 - danger_level * 80 + math.sin(elapsed) * 10)
                    stability_before = max(10, 100 - danger_level * 90 + math.sin(elapsed * 2) * 5)
                    battery_before = max(20, 100 - elapsed * 3)
                else:
                    alt_before = 0
                    stability_before = 0
                    battery_before = 0
                        
                self.before_canvas.create_text(width_before-90, 55,
                                             text=f"ALT: {alt_before:.0f}m",
                                             fill="#ff6666", font=("Courier", 11))
                self.before_canvas.create_text(width_before-90, 75,
                                             text=f"STAB: {stability_before:.0f}%",
                                             fill="#ff6666", font=("Courier", 11))
                self.before_canvas.create_text(width_before-90, 95,
                                             text=f"BATT: {battery_before:.0f}%",
                                             fill="#ff6666", font=("Courier", 11))
                
            if phase >= 3:
                # After drone telemetry (stable/improving)
                alt_after = 100 + 50 * altitude_factor + math.sin(elapsed) * 3
                stability_after = 95 + 5 * math.sin(elapsed * 0.5)
                battery_after = max(30, 100 - elapsed * 0.5)
                    
                self.after_canvas.create_text(width_after-90, 55,
                                            text=f"ALT: {alt_after:.0f}m",
                                            fill="#66ff66", font=("Courier", 11))
                self.after_canvas.create_text(width_after-90, 75,
                                            text=f"STAB: {stability_after:.0f}%",
                                            fill="#66ff66", font=("Courier", 11))
                self.after_canvas.create_text(width_after-90, 95,
                                            text=f"BATT: {battery_after:.0f}%",
                                            fill="#66ff66", font=("Courier", 11))
                    
        except Exception as e:
            print(f"Extended canvas update error: {e}")
            pass

def main():
    """Main function for mission-critical demo"""
    root = tk.Tk()
    app = RevolutionaryDroneDemo(root)
    
    # Clean window positioning
    root.update_idletasks()
    x = (root.winfo_screenwidth() // 2) - (1200 // 2)
    y = (root.winfo_screenheight() // 2) - (800 // 2)
    root.geometry(f"1200x800+{x}+{y}")
    
    # Handle cleanup
    def on_closing():
        app.mission_active = False
        root.destroy()
    
    root.protocol("WM_DELETE_WINDOW", on_closing)
    root.mainloop()

if __name__ == "__main__":
    main()