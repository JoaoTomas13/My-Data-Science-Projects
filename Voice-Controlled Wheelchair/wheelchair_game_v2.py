####################################################################
#                                                                  #
#                      Wheelchair Game                             #
#                                                                  #
#   control a wheelchair through a maze using voice commands.      #
#   Make the blue arrow reach the green square for winning.        #
#   WASD + space keys available for controlling the wheelchair     #
#   simultaneously with the voice commands.                        #
#                                                                  #
#   Libraries needed: pygame, sounddevice, numpy                   #
#                                                                  #
#     Developed by Marco Simoes (msimoes@dei.uc.pt) - DEI 2024     #
#                                                                  #
####################################################################


import pygame
import random
import math
import sounddevice as sd
import numpy as np
import threading
import time
import matplotlib.pyplot as plt

# init pygame
pygame.init()

# define constants
SCREEN_WIDTH, SCREEN_HEIGHT = 855, 675
CELL_SIZE = 15  # cell size
CELL_EXPANSION = 3  # number of cells within each cell
FPS = 10
MAZE_ROWS = SCREEN_HEIGHT // CELL_SIZE // CELL_EXPANSION  # number of rows
MAZE_COLS = SCREEN_WIDTH // CELL_SIZE // CELL_EXPANSION # number of cols
INITIAL_SCORE = 0  # counter of crashes

# colors
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
BLUE = (0, 0, 255)
RED = (255, 0, 0)
GREEN = (0, 255, 0)


# game screen set up
screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
pygame.display.set_caption("Wheelchair Labyrinth")

last_classification_time = time.time()


class Maze:
    ''' Maze class. Sets up and draws the maze.'''
    
    def __init__(self):
        # base grid
        self.base_grid = [[1 for _ in range(MAZE_COLS)] for _ in range(MAZE_ROWS)]
        # expanded grid
        self.expanded_grid = [[1 for _ in range(MAZE_COLS * CELL_EXPANSION)] for _ in range(MAZE_ROWS * CELL_EXPANSION)]
        
        self.generate_maze()
        self.expand_maze()
        
        # place the target in a randow white cell
        self.target = self.get_random_white_cell()

    def generate_maze(self):
        # simple DFS algorithm to build the maze
        
        stack = [(1, 1)]
        self.base_grid[1][1] = 0

        while stack:
            row, col = stack[-1]
            neighbors = []

            if row > 1 and self.base_grid[row - 2][col] == 1:
                neighbors.append((row - 2, col))
            if row < MAZE_ROWS - 2 and self.base_grid[row + 2][col] == 1:
                neighbors.append((row + 2, col))
            if col > 1 and self.base_grid[row][col - 2] == 1:
                neighbors.append((row, col - 2))
            if col < MAZE_COLS - 2 and self.base_grid[row][col + 2] == 1:
                neighbors.append((row, col + 2))

            if neighbors:
                next_row, next_col = random.choice(neighbors)
                self.base_grid[next_row][next_col] = 0
                self.base_grid[(next_row + row) // 2][(next_col + col) // 2] = 0
                stack.append((next_row, next_col))
            else:
                stack.pop()

    def expand_maze(self):
        # expand rows so each place is a CELL_EXPANSION by CELL_EXPANSION mini-grid of places
        for row in range(MAZE_ROWS):
            for col in range(MAZE_COLS):
                if self.base_grid[row][col] == 0:
                    # expand cells
                    for i in range(CELL_EXPANSION):
                        for j in range(CELL_EXPANSION):
                            self.expanded_grid[row * CELL_EXPANSION + i][col * CELL_EXPANSION + j] = 0

    def get_random_white_cell(self):
        # find random white cell
        while True:
            row = random.randint(0, MAZE_ROWS-1)
            col = random.randint(0, MAZE_COLS-1)
            if self.expanded_grid[int((row+0.5)*CELL_EXPANSION)][int((col+0.5)*CELL_EXPANSION)] == 0:
                return (int((col+0.5)*CELL_EXPANSION), int((row+0.5)*CELL_EXPANSION))

    def draw(self):
        # draw the maze
        for row in range(MAZE_ROWS * CELL_EXPANSION):
            for col in range(MAZE_COLS * CELL_EXPANSION):
                color = WHITE if self.expanded_grid[row][col] == 0 else BLACK
                pygame.draw.rect(screen, color, (col * CELL_SIZE, row * CELL_SIZE, CELL_SIZE, CELL_SIZE))

        # draw the target
        target_x, target_y = self.target
        pygame.draw.rect(screen, GREEN, (target_x * CELL_SIZE, target_y * CELL_SIZE, CELL_SIZE, CELL_SIZE))


class Wheelchair:
    '''Wheelchair class. Controls the wheelchair and game dynamics.'''
    
    def __init__(self, maze):
        self.x, self.y = 1.5*CELL_SIZE*CELL_EXPANSION, 1.5*CELL_SIZE*CELL_EXPANSION
        self.direction = 0  # angle in degrees (0 is right)
        self.speed = 0
        self.maze = maze  
        self.score = INITIAL_SCORE  # initial score
        self.win = False  # win flag

    def can_move(self, dx, dy):
        # compute new position
        new_x = (self.x + dx) // CELL_SIZE
        new_y = (self.y + dy) // CELL_SIZE

        # check if new position is white and whithin the board
        if 0 <= new_x < len(self.maze.expanded_grid[0]) and 0 <= new_y < len(self.maze.expanded_grid):
            return self.maze.expanded_grid[int(new_y)][int(new_x)] == 0
        return False

    def move(self):
        if self.win:
            return  # after win, disallow movement

        # compute dx and dy based on the direction (angle in degrees)
        rad = math.radians(self.direction)
        dx = self.speed * math.cos(rad)
        dy = -self.speed * math.sin(rad)

        # check if can move before changing position
        if self.can_move(dx, dy):
            self.x += dx
            self.y += dy
        else:
            # count collision
            self.score += 1
            self.speed = 0

    def rotate(self, direction):
        if self.win:
            return  # stop rotation after winning
        if direction == "LEFT":
            self.direction = (self.direction + 45) % 360
        elif direction == "RIGHT":
            self.direction = (self.direction - 45) % 360

    def stop(self):
        self.speed = 0

    def execute_command(self, command):
        if command == "FORWARD":
            self.speed = CELL_SIZE // 2
        elif command == "BACKWARD":
            self.speed = -CELL_SIZE // 2
        elif command == "LEFT":
            self.rotate("LEFT")
        elif command == "RIGHT":
            self.rotate("RIGHT")
        elif command == "STOP":
            self.stop()

    def check_win(self):
        # check if target is reached
        target_x, target_y = self.maze.target
        return (self.x // CELL_SIZE, self.y // CELL_SIZE) == (target_x, target_y)

    def draw(self):
        # draw wheelchair as a triangle
        rad = math.radians(self.direction)
        front_x = self.x + CELL_SIZE * math.cos(rad)
        front_y = self.y - CELL_SIZE * math.sin(rad)
        left_x = self.x + CELL_SIZE * math.cos(rad + math.radians(135))
        left_y = self.y - CELL_SIZE * math.sin(rad + math.radians(135))
        right_x = self.x + CELL_SIZE * math.cos(rad - math.radians(135))
        right_y = self.y - CELL_SIZE * math.sin(rad - math.radians(135))

        pygame.draw.polygon(screen, BLUE, [(front_x, front_y), (left_x, left_y), (right_x, right_y)]) 


def sound_capture_thread(wheelchair):
    '''thread for capturing the microphone sound and converting it to a command to the wheelchair.'''
    
    RATE = 16000  # sampling rate of 16kHz
    RECORD_SECONDS = 1  # capture 1 second of sound
    OVERLAP_SECONDS = .95  # overlap between segments
    NO_OVERLAP_SECONDS = RECORD_SECONDS - OVERLAP_SECONDS # new data on each segment

    AMP_THRESHOLD = 0.2  # minimum amplitude to consider sound
    GAP_TIME_THRESHOLD = 0.5  # minimum time between classifications

    buffer_size = int(RATE * RECORD_SECONDS)
    buffer = np.zeros(buffer_size, dtype=np.float32)

    def callback(indata, frames, timestamp, status):
        global last_classification_time
        nonlocal buffer
        if status:
            print(status)
        
        # add the new data to the buffer, rolling the current data to the left
        buffer = np.roll(buffer, -len(indata))
        buffer[-len(indata):] = indata[:, 0]

        center_buffer = buffer[len(buffer)//2 - len(buffer)//10 : len(buffer)//2 + len(buffer)//10]
        if np.max(np.abs(center_buffer)) < AMP_THRESHOLD or (time.time() - last_classification_time) < GAP_TIME_THRESHOLD:
            return
        
        last_classification_time = time.time()
        
        # convert the captured sound to a command
        command = process_sound(buffer)

        if command:
            wheelchair.execute_command(command)

    # open audio stream from microphone
    with sd.InputStream(device=None, callback=callback, channels=1, samplerate=RATE, blocksize=int(RATE * NO_OVERLAP_SECONDS)):
        while True:
            time.sleep(NO_OVERLAP_SECONDS)  # Aguardar NO_OVERLAP_SECONDS


def main():
    '''Main function of the game.'''
    global start_recording_flag, start_recording_time
    
    clock = pygame.time.Clock()
    maze = Maze()
    wheelchair = Wheelchair(maze)
    running = True
    start_ticks = pygame.time.get_ticks()  # Tempo inicial

    # start sound capture thread
    sound_thread = threading.Thread(target=sound_capture_thread, args=(wheelchair,))
    sound_thread.daemon = True
    sound_thread.start()

    while running:
        screen.fill(WHITE)
        maze.draw()
        wheelchair.draw()

        # render points on screen
        font = pygame.font.SysFont(None, 36)
        score_text = font.render(f"Colisions: {wheelchair.score}", True, RED)
        screen.blit(score_text, (10, 10))

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if wheelchair.win:
                    continue  # stop wheelchair control after winning
                if event.key == pygame.K_w:  # Forward
                    wheelchair.execute_command("FORWARD")
                elif event.key == pygame.K_s:  # Backward
                    wheelchair.execute_command("BACKWARD")
                elif event.key == pygame.K_a:  # Rotate left (45 degrees)
                    wheelchair.execute_command("LEFT")
                elif event.key == pygame.K_d:  # Rotate right (45 degrees)
                    wheelchair.execute_command("RIGHT")
                elif event.key == pygame.K_SPACE:  # Stop
                    wheelchair.execute_command("STOP")
                


        # check victory
        if wheelchair.check_win():
            wheelchair.win = True  # register victory
            font = pygame.font.SysFont(None, 44)
            win_text = font.render("Congratulations! You reached the target location!", True, GREEN)
            screen.blit(win_text, (SCREEN_WIDTH // 15, SCREEN_HEIGHT // 2))
            wheelchair.stop()  # stop the wheelchair

        wheelchair.move()

        pygame.display.flip()
        clock.tick(FPS)

    pygame.quit()


###############################################################################################
#                                                                                             #
#                              STUDENT CODE STARTS HERE                                       #
#                                                                                             #
# Edit, replace, change the following code as you wish. The only thing that                   #
# must be kept is the name of the function process_sound, and it must return                  #
# a string with a command for the game to act. The list of possible commands is:              #
# ["FORWARD", "BACKWARD", "LEFT", "RIGHT", "STOP", ""] -> empty string means do nothing       #
#                                                                                             #
###############################################################################################
from joblib import load

commands = { 'forward': "FORWARD", 'backward': "BACKWARD", 'left': "LEFT", 'right': "RIGHT", 'stop': "STOP" , '_unknown_':"UNKNOWN", '_silence_':"SILENCE"}


# load classifier from file svm saved in Module B
classifier = load('svm_model1.pkl')
scaler = load("scaler.pkl")


def classify(features):
    '''Receives a feature vector and returns a prediction from the classifier.'''
    features = np.array(features).reshape(1, -1)
    prediction = classifier.predict(features)
    
    # Debugging
    print(f"Classifier prediction: {prediction[0]}")
    
    # Map prediction to command
    try:
        return commands[prediction[0]]
    except KeyError:
        print(f"Unexpected prediction value: {prediction[0]}")
        return ""

import librosa
import numpy as np
import pandas as pd

import librosa
import numpy as np
import pandas as pd


def extrair_caracteristicas(audio_data, samplerate, segment_duration_ms=200, overlap_percentage=0.5):
    '''Recebe o áudio completo e retorna um vetor de características extraídas com sobreposição.'''

    # Checar se o áudio tem comprimento mínimo
    if len(audio_data) < 1000:
        raise ValueError("O áudio é muito curto para análise. Verifique o tamanho do áudio.")

    # Calcular número de amostras por segmento
    segment_duration_samples = int(samplerate * (segment_duration_ms / 1000))
    step_size = int(segment_duration_samples * (1 - overlap_percentage))  # Quantidade de avanço em cada iteração

    features = []

    # Iterar sobre os segmentos com sobreposição
    for start_idx in range(0, len(audio_data) - segment_duration_samples + 1, step_size):
        end_idx = start_idx + segment_duration_samples
        segment = audio_data[start_idx:end_idx]

        if len(segment) < segment_duration_samples:
            continue

        # Lista para armazenar as características do segmento
        segment_features = []

        # 1. MFCCs (25 características)
        mfccs = librosa.feature.mfcc(y=segment, sr=samplerate, n_mfcc=25, n_fft=512, hop_length=256)
        segment_features.extend(np.mean(mfccs, axis=1))  # Média dos MFCCs

        # 2. Spectral Contrast (3 características)
        spectral_contrast = librosa.feature.spectral_contrast(y=segment, sr=samplerate, n_fft=512, hop_length=256)
        segment_features.extend(np.mean(spectral_contrast[:, :3], axis=1))  # Pegando apenas as primeiras 3 médias

        # 3. Zero Crossing Rate (1 característica)
        zero_crossings = librosa.feature.zero_crossing_rate(y=segment)
        segment_features.append(float(np.mean(zero_crossings)))

        # 4. Roll-off Spectral (1 característica)
        rolloff = librosa.feature.spectral_rolloff(y=segment, sr=samplerate, n_fft=512, hop_length=256)
        segment_features.append(float(np.mean(rolloff)))

        # 5. Largura de Banda Espectral (1 característica)
        bandwidth = librosa.feature.spectral_bandwidth(y=segment, sr=samplerate, n_fft=512, hop_length=256)
        segment_features.append(float(np.mean(bandwidth)))

        # 6. Energia RMS (1 característica)
        rms = librosa.feature.rms(y=segment)
        segment_features.append(float(np.mean(rms)))

        # 7. Pitch (1 característica usando análise de harmônicos)
        pitches, magnitudes = librosa.core.piptrack(y=segment, sr=samplerate)
        if pitches.size > 0:
            pitch_mean = np.mean(pitches[pitches > 0])  # Média apenas de frequências válidas
        else:
            pitch_mean = 0
        segment_features.append(float(pitch_mean))

        # 8. Centroide Espectral (1 característica)
        spectral_centroid = librosa.feature.spectral_centroid(y=segment, sr=samplerate, n_fft=512, hop_length=256)
        segment_features.append(float(np.mean(spectral_centroid)))

        # 9. Contraste Espectral (3 características)
        spectral_contrast = librosa.feature.spectral_contrast(y=segment, sr=samplerate, n_fft=512, hop_length=256)
        segment_features.extend(np.mean(spectral_contrast[:, :3], axis=1))  # Obter as primeiras 3 médias

        # Adicionar segment_features ao vetor principal
        features.extend(segment_features)

    

    # Validar que temos o vetor com tamanho esperado
    if len(features) < 39:
        raise ValueError(f"O vetor de características tem menos dimensões do que o esperado. Atual: {len(features)}")
    
    # Ajustar para 25 características se necessário
    if len(features) >= 39:
        features = features[:39]  # Mantém apenas os primeiros 25 valores
    elif len(features) < 39:
        raise ValueError("O vetor de características tem menos dimensões que o esperado pelo modelo.")
    
    # Substituir NaNs por zero
    features = np.nan_to_num(features, nan=0.0)  # Tratar NaNs substituindo por 0

    # Exibir para debug
    print("\nCaracterísticas extraídas:")
    print(pd.DataFrame(features).T)

    return np.array(features)



def process_sound(sound):
    ''' receives a 1-second sound segment from the microphone and returns a command '''
    command = ""
    
    print(f"Formato do som: {type(sound)}, tamanho: {len(sound)}, valores: {sound[:10]}")
    feature_vector = extrair_caracteristicas(sound, 16000, segment_duration_ms=200, overlap_percentage=0.5)

    command = classify(feature_vector)
    
    return command



if __name__ == "__main__":
    main()