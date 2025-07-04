# Video Sampling Methods

## Sampler Methods
- **Required Parameters**: `mode`: select one from (`uniform`,`head`,`group`,`slide`)

### 1. Uniform Sampling `uniform`
- **Description**: Selects frames evenly spaced throughout the video
- **Extra Required Parameters**:
  - `num_frames`: Total number of frames to sample(int)

### 2. Head Sampling `head`
- **Description**: Selects frames from the beginning of the video
- **Extra Required Parameters**:
  - `num_frames`: Number of frames to take from the start(int)

### 3. Group Sampling `group`
- **Description**: Divides video into segments and samples frames from each
- **Extra Required Parameters**:
  - `num_groups`: Number of segments to divide the video into(int)
  - `frames_per_group`: Frames to sample from each segment(int)

### 4. Slide Sampling `slide`
- **Description**: Uses a sliding window approach to sample frames
- **Extra Required Parameters**:
  - `window_size`: Number of frames in each window(int)
  - `stride`: How many frames to move between windows(int)

## Frame Strategies for Short Videos
- **Required Parameters**: `frame_strategy`: select one from (`zero_pad`,`discard`)

### 1. Zero Pad `zero_pad`
- **Description**: Pad sampled frames shorter than the required frame count with zero frames

### 2. Discard `discard`
- **Description**: Drop sampled frames shorter than the required frame count
