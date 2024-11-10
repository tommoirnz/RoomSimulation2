import tkinter as tk
from tkinter import filedialog, messagebox
import pyroomacoustics as pra
import soundfile as sf
import numpy as np
import os
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg  # Import for embedding plots in Tkinter

class ConvolutionMixerGUI:
    def __init__(self, master):
        self.master = master
        master.title("Stereo Convolution Mixer with Debugging")

        # Initialize variables
        self.input_file = None
        self.output_file = None
        self.input_properties = {}
        self.output_properties = {}

        # ----------- Input File Selection -----------
        input_frame = tk.LabelFrame(master, text="Input Audio File", padx=10, pady=10)
        input_frame.grid(row=0, column=0, padx=10, pady=10, sticky='ew')

        self.label_input = tk.Label(input_frame, text="No input file selected.")
        self.label_input.grid(row=0, column=0, columnspan=3, sticky='w')

        self.button_browse = tk.Button(input_frame, text="Browse Audio File", command=self.browse_file)
        self.button_browse.grid(row=0, column=3, padx=5, pady=5)

        # Input File Properties
        self.label_input_props = tk.Label(input_frame, text="File Properties:")
        self.label_input_props.grid(row=1, column=0, columnspan=4, sticky='w', pady=(10,0))

        self.input_props_text = tk.Text(input_frame, height=5, width=80, state='disabled')
        self.input_props_text.grid(row=2, column=0, columnspan=4, pady=5)

        # ----------- Room Parameters -----------
        room_frame = tk.LabelFrame(master, text="Room Parameters (meters)", padx=10, pady=10)
        room_frame.grid(row=1, column=0, padx=10, pady=10, sticky='ew')

        tk.Label(room_frame, text="Length:").grid(row=0, column=0, padx=5, pady=5, sticky='e')
        self.entry_length = tk.Entry(room_frame, width=10)
        self.entry_length.insert(0, "10")
        self.entry_length.grid(row=0, column=1, padx=5, pady=5)

        tk.Label(room_frame, text="Width:").grid(row=0, column=2, padx=5, pady=5, sticky='e')
        self.entry_width = tk.Entry(room_frame, width=10)
        self.entry_width.insert(0, "7")
        self.entry_width.grid(row=0, column=3, padx=5, pady=5)

        tk.Label(room_frame, text="Height:").grid(row=0, column=4, padx=5, pady=5, sticky='e')
        self.entry_height = tk.Entry(room_frame, width=10)
        self.entry_height.insert(0, "3")
        self.entry_height.grid(row=0, column=5, padx=5, pady=5)

        tk.Label(room_frame, text="Absorption Coefficient (0-1):").grid(row=1, column=0, padx=5, pady=5, sticky='e')
        self.entry_absorption = tk.Entry(room_frame, width=10)
        self.entry_absorption.insert(0, "0.2")
        self.entry_absorption.grid(row=1, column=1, padx=5, pady=5)

        # ----------- Source and Microphone Positions -----------
        positions_frame = tk.LabelFrame(master, text="Source and Microphone Positions (meters)", padx=10, pady=10)
        positions_frame.grid(row=2, column=0, padx=10, pady=10, sticky='ew')

        # Source 1 (Front-Left Corner)
        tk.Label(positions_frame, text="Source 1 (Front-Left) [x, y, z]:").grid(row=0, column=0, padx=5, pady=5, sticky='e')
        self.entry_s1 = tk.Entry(positions_frame, width=20)
        self.entry_s1.insert(0, "0, 0, 0")  # Front-Left corner at ground level
        self.entry_s1.grid(row=0, column=1, padx=5, pady=5)

        # Source 2 (Front-Right Corner)
        tk.Label(positions_frame, text="Source 2 (Front-Right) [x, y, z]:").grid(row=0, column=2, padx=5, pady=5, sticky='e')
        self.entry_s2 = tk.Entry(positions_frame, width=20)
        self.entry_s2.insert(0, "0, 7, 0")  # Front-Right corner at ground level
        self.entry_s2.grid(row=0, column=3, padx=5, pady=5)

        # Microphone Positions
        # Microphone Left
        tk.Label(positions_frame, text="Microphone Left [x, y, z]:").grid(row=1, column=0, padx=5, pady=5, sticky='e')
        self.entry_mic_left = tk.Entry(positions_frame, width=20)
        # Positioning microphones 0.2m apart along the y-axis and at 1m height
        # Assuming center of room is at (5, 3.5, 1)
        mic_left_pos = [5, 3.5 - 0.1, 1.0]  # 0.1m to the left of center
        self.entry_mic_left.insert(0, f"{mic_left_pos[0]}, {mic_left_pos[1]}, {mic_left_pos[2]}")
        self.entry_mic_left.grid(row=1, column=1, padx=5, pady=5)

        # Microphone Right
        tk.Label(positions_frame, text="Microphone Right [x, y, z]:").grid(row=1, column=2, padx=5, pady=5, sticky='e')
        self.entry_mic_right = tk.Entry(positions_frame, width=20)
        mic_right_pos = [5, 3.5 + 0.1, 1.0]  # 0.1m to the right of center
        self.entry_mic_right.insert(0, f"{mic_right_pos[0]}, {mic_right_pos[1]}, {mic_right_pos[2]}")
        self.entry_mic_right.grid(row=1, column=3, padx=5, pady=5)

        # ----------- Output File Selection -----------
        output_frame = tk.LabelFrame(master, text="Output Audio File", padx=10, pady=10)
        output_frame.grid(row=3, column=0, padx=10, pady=10, sticky='ew')

        self.label_output = tk.Label(output_frame, text="No output file selected.")
        self.label_output.grid(row=0, column=0, columnspan=3, sticky='w')

        self.button_save = tk.Button(output_frame, text="Save Output File", command=self.save_file)
        self.button_save.grid(row=0, column=3, padx=5, pady=5)

        # Output File Properties
        self.label_output_props = tk.Label(output_frame, text="File Properties:")
        self.label_output_props.grid(row=1, column=0, columnspan=4, sticky='w', pady=(10,0))

        self.output_props_text = tk.Text(output_frame, height=5, width=80, state='disabled')
        self.output_props_text.grid(row=2, column=0, columnspan=4, pady=5)

        # ----------- Process Button -----------
        self.button_process = tk.Button(master, text="Process and Convolve", command=self.process_audio, bg="green", fg="white", font=('Helvetica', 12, 'bold'))
        self.button_process.grid(row=4, column=0, padx=10, pady=20, sticky='ew')

    def browse_file(self):
        filetypes = (
            ("WAV files", "*.wav"),
            ("All files", "*.*")
        )
        filename = filedialog.askopenfilename(title="Open Audio File", initialdir=os.getcwd(), filetypes=filetypes)
        if filename:
            self.input_file = filename
            self.label_input.config(text=f"Input File: {os.path.basename(filename)}")
            self.display_input_properties()

    def save_file(self):
        filetypes = (
            ("WAV files", "*.wav"),
            ("All files", "*.*")
        )
        filename = filedialog.asksaveasfilename(title="Save Mixed Audio", defaultextension=".wav", filetypes=filetypes)
        if filename:
            self.output_file = filename
            self.label_output.config(text=f"Output File: {os.path.basename(filename)}")
            # Clear previous output properties
            self.clear_output_properties()

    def process_audio(self):
        if not self.input_file:
            messagebox.showerror("Error", "Please select an input audio file.")
            return
        if not self.output_file:
            messagebox.showerror("Error", "Please specify an output file.")
            return

        try:
            # Read the input stereo audio file
            speech, fs = sf.read(self.input_file)
            input_info = sf.info(self.input_file)
            print(f"Input Audio Info: {input_info}")
            if speech.ndim != 2 or speech.shape[1] != 2:
                messagebox.showerror("Error", "Input audio must be a stereo file with two channels.")
                return

            # Room parameters
            length = float(self.entry_length.get())
            width = float(self.entry_width.get())
            height = float(self.entry_height.get())
            absorption = float(self.entry_absorption.get())
            if not (0 <= absorption <= 1):
                messagebox.showerror("Error", "Absorption coefficient must be between 0 and 1.")
                return

            room_dim = [length, width, height]
            print(f"Room Dimensions: {room_dim}")
            print(f"Absorption Coefficient: {absorption}")

            # Source positions
            s1_pos = self.parse_position(self.entry_s1.get())
            s2_pos = self.parse_position(self.entry_s2.get())
            if s1_pos is None or s2_pos is None:
                messagebox.showerror("Error", "Invalid source positions.")
                return
            print(f"Source 1 Position: {s1_pos}")
            print(f"Source 2 Position: {s2_pos}")

            # Microphone positions
            mic_left_pos = self.parse_position(self.entry_mic_left.get())
            mic_right_pos = self.parse_position(self.entry_mic_right.get())
            if mic_left_pos is None or mic_right_pos is None:
                messagebox.showerror("Error", "Invalid microphone positions.")
                return
            print(f"Microphone Left Position: {mic_left_pos}")
            print(f"Microphone Right Position: {mic_right_pos}")

            # Create the room
            room = pra.ShoeBox(room_dim, fs=fs, absorption=absorption, max_order=17)
            print("Room created.")

            # Add sources
            room.add_source(s1_pos, signal=speech[:,0])
            room.add_source(s2_pos, signal=speech[:,1])
            print("Sources added.")

            # Add microphone array (two microphones)
            room.add_microphone_array(np.array([mic_left_pos, mic_right_pos]).T)
            print("Microphone array added.")

            # Compute the room impulse responses
            room.compute_rir()
            print("Impulse responses computed.")

            # Simulate the room (convolve sources with RIRs)
            room.simulate()
            print("Room simulation completed.")

            # Retrieve the mixed signals for both microphones
            mixed_left = room.mic_array.signals[0]
            mixed_right = room.mic_array.signals[1]
            print(f"Mixed Left Signal: min={mixed_left.min()}, max={mixed_left.max()}")
            print(f"Mixed Right Signal: min={mixed_right.min()}, max={mixed_right.max()}")

            # Check if signals have data
            if np.all(mixed_left == 0) and np.all(mixed_right == 0):
                messagebox.showerror("Error", "Mixed signals are silent. Check source and microphone positions.")
                return

            # Stack into stereo
            mixed_stereo = np.vstack((mixed_left, mixed_right)).T
            print(f"Mixed Stereo Shape: {mixed_stereo.shape}")

            # Normalize the mixed signal to prevent clipping
            max_val = np.max(np.abs(mixed_stereo))
            print(f"Maximum value before normalization: {max_val}")
            if max_val > 0:
                mixed_stereo /= max_val
                print("Signals normalized.")

            # Save the mixed stereo audio
            sf.write(self.output_file, mixed_stereo, fs)
            print(f"Mixed audio saved to {self.output_file}")

            # After saving, display output properties
            self.display_output_properties()

            # Plot the room layout in separate Tkinter windows
            self.plot_plan_view(room_dim, [s1_pos, s2_pos], [mic_left_pos, mic_right_pos])
            self.plot_end_view(room_dim, [s1_pos, s2_pos], [mic_left_pos, mic_right_pos])

            messagebox.showinfo("Success", f"Mixed stereo audio saved as:\n{self.output_file}")

        except Exception as e:
            messagebox.showerror("Error", f"An error occurred:\n{str(e)}")
            print(f"Error: {e}")

    def parse_position(self, pos_str):
        try:
            parts = pos_str.split(',')
            if len(parts) != 3:
                return None
            pos = [float(part.strip()) for part in parts]
            return pos
        except:
            return None

    def display_input_properties(self):
        try:
            info = sf.info(self.input_file)
            props = {
                "File Name": os.path.basename(self.input_file),
                "Channels": "Stereo" if info.channels == 2 else "Mono",
                "Sampling Rate": f"{info.samplerate} Hz",
                "Bit Depth": info.subtype,  # e.g., 'PCM_16'
                "Duration": f"{info.duration:.2f} seconds"
            }
            self.input_properties = props
            self.input_props_text.config(state='normal')
            self.input_props_text.delete(1.0, tk.END)
            for key, value in props.items():
                self.input_props_text.insert(tk.END, f"{key}: {value}\n")
            self.input_props_text.config(state='disabled')
        except Exception as e:
            messagebox.showerror("Error", f"Failed to read input file properties:\n{str(e)}")
            print(f"Error reading input properties: {e}")

    def display_output_properties(self):
        try:
            info = sf.info(self.output_file)
            duration = info.frames / info.samplerate
            props = {
                "File Name": os.path.basename(self.output_file),
                "Channels": "Stereo" if info.channels == 2 else "Mono",
                "Sampling Rate": f"{info.samplerate} Hz",
                "Bit Depth": info.subtype,  # e.g., 'PCM_16'
                "Duration": f"{duration:.2f} seconds"
            }
            self.output_properties = props
            self.output_props_text.config(state='normal')
            self.output_props_text.delete(1.0, tk.END)
            for key, value in props.items():
                self.output_props_text.insert(tk.END, f"{key}: {value}\n")
            self.output_props_text.config(state='disabled')
        except Exception as e:
            messagebox.showerror("Error", f"Failed to read output file properties:\n{str(e)}")
            print(f"Error reading output properties: {e}")

    def clear_output_properties(self):
        self.output_props_text.config(state='normal')
        self.output_props_text.delete(1.0, tk.END)
        self.output_props_text.config(state='disabled')

    def plot_plan_view(self, room_dim, sources, microphones):
        """
        Plots a top-down (plan) view of the room with sources and microphones
        in a separate Tkinter window.
        Sources are marked with 'X' and microphones with 'O'.
        """
        try:
            length, width, height = room_dim

            # Create a new Toplevel window
            plot_window = tk.Toplevel(self.master)
            plot_window.title("Room Layout - Plan View")

            # Create a Matplotlib figure
            fig, ax = plt.subplots(figsize=(8, 6))

            # Draw the room boundaries
            room_rect = plt.Rectangle((0, 0), width, length, fill=None, edgecolor='black', linewidth=2)
            ax.add_patch(room_rect)

            # Plot sources
            ax.scatter(
                [source[1] for source in sources],
                [source[0] for source in sources],
                marker='x',
                color='red',
                s=100,
                label='Sources'
            )

            # Plot microphones
            ax.scatter(
                [mic[1] for mic in microphones],
                [mic[0] for mic in microphones],
                marker='o',
                color='blue',
                s=100,
                label='Microphones'
            )

            # Set labels and title
            ax.set_xlabel('Width (meters)')
            ax.set_ylabel('Length (meters)')
            ax.set_title('Room Layout - Plan View')
            ax.set_xlim(-1, width + 1)
            ax.set_ylim(-1, length + 1)
            ax.grid(True)

            # Create a legend with only two entries: Sources and Microphones
            ax.legend(loc='upper right')

            ax.set_aspect('equal', adjustable='box')

            # Embed the plot in the Tkinter window
            canvas = FigureCanvasTkAgg(fig, master=plot_window)
            canvas.draw()
            canvas.get_tk_widget().pack(fill=tk.BOTH, expand=1)

        except Exception as e:
            messagebox.showerror("Error", f"Failed to plot plan view:\n{str(e)}")
            print(f"Error plotting plan view: {e}")

    def plot_end_view(self, room_dim, sources, microphones):
        """
        Plots an end (side) view of the room with sources and microphones
        in a separate Tkinter window.
        Sources are marked with 'X' and microphones with 'O'.
        """
        try:
            length, width, height = room_dim

            # Choose which axis to use for the end view.
            # For this example, we'll use the x (length) vs z (height) axes.
            # Alternatively, you could use y vs z based on preference.

            # Create a new Toplevel window
            plot_window = tk.Toplevel(self.master)
            plot_window.title("Room Layout - End View")

            # Create a Matplotlib figure
            fig, ax = plt.subplots(figsize=(8, 6))

            # Draw the room boundaries
            room_rect = plt.Rectangle((0, 0), length, height, fill=None, edgecolor='black', linewidth=2)
            ax.add_patch(room_rect)

            # Plot sources
            ax.scatter(
                [source[0] for source in sources],  # x-axis
                [source[2] for source in sources],  # z-axis
                marker='x',
                color='red',
                s=100,
                label='Sources'
            )

            # Plot microphones
            ax.scatter(
                [mic[0] for mic in microphones],    # x-axis
                [mic[2] for mic in microphones],    # z-axis
                marker='o',
                color='blue',
                s=100,
                label='Microphones'
            )

            # Set labels and title
            ax.set_xlabel('Length (meters)')
            ax.set_ylabel('Height (meters)')
            ax.set_title('Room Layout - End View')
            ax.set_xlim(-1, length + 1)
            ax.set_ylim(-1, height + 1)
            ax.grid(True)

            # Create a legend with only two entries: Sources and Microphones
            ax.legend(loc='upper right')

            ax.set_aspect('equal', adjustable='box')

            # Embed the plot in the Tkinter window
            canvas = FigureCanvasTkAgg(fig, master=plot_window)
            canvas.draw()
            canvas.get_tk_widget().pack(fill=tk.BOTH, expand=1)

        except Exception as e:
            messagebox.showerror("Error", f"Failed to plot end view:\n{str(e)}")
            print(f"Error plotting end view: {e}")

def main():
    root = tk.Tk()
    gui = ConvolutionMixerGUI(root)
    root.mainloop()

if __name__ == "__main__":
    main()
