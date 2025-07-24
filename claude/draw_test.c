#include <SDL2/SDL.h>
#include <SDL2/SDL_ttf.h>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <stdbool.h>

#include "nn.h"

#define WIDTH 280
#define HEIGHT 280
#define CELL_SIZE (WIDTH / 28)

#define INPUT 784
#define OUTPUT 10

SDL_Window *window = NULL;
SDL_Renderer *renderer = NULL;
TTF_Font *font = NULL;
SDL_Color text_color = {255, 255, 255, 255};
float pixels[28][28] = {0};

// Neural Network Functions
Network* load_network(const char *filename) {
    FILE *file = fopen(filename, "rb");
    if (!file) {
        printf("Could not open weights file: %s\n", filename);
        return NULL;
    }
    
    int num_layers;
    fread(&num_layers, sizeof(int), 1, file);
    
    int *sizes = malloc((num_layers + 1) * sizeof(int));
    fread(sizes, sizeof(int), num_layers + 1, file);
    
    Network *net = malloc(sizeof(Network));
    net->num_layers = num_layers;
    net->layers = malloc(num_layers * sizeof(Layer));
    
    // Initialize layers
    for (int i = 0; i < num_layers; i++) {
        if (!init_layer(&net->layers[i], sizes[i], sizes[i + 1])) {
            printf("Failed to initialize layer %d\n", i);
            free(sizes);
            fclose(file);
            return NULL;
        }
    }
    
    // Load weights and biases
    for (int i = 0; i < num_layers; i++) {
        Layer *l = &net->layers[i];
        fread(l->biases, sizeof(float), l->size, file);
        
        for (int j = 0; j < l->size; j++) {
            fread(l->weights[j], sizeof(float), l->pre_size, file);
        }
    }
    
    free(sizes);
    fclose(file);
    
    printf("Network loaded successfully!\n");
    printf("Architecture: %d", INPUT);
    for (int i = 0; i < net->num_layers; i++) {
        printf(" -> %d", net->layers[i].size);
    }
    printf("\n");
    
    return net;
}

// Drawing functions
void draw_grid() {
    SDL_SetRenderDrawColor(renderer, 50, 50, 50, 255);
    for (int i = 0; i <= 28; i++) {
        SDL_RenderDrawLine(renderer, i*CELL_SIZE, 0, i*CELL_SIZE, HEIGHT);
        SDL_RenderDrawLine(renderer, 0, i*CELL_SIZE, WIDTH, i*CELL_SIZE);
    }
}

void draw_pixels() {
    for (int y = 0; y < 28; y++) {
        for (int x = 0; x < 28; x++) {
            int color = (int)(pixels[y][x] * 255);
            SDL_SetRenderDrawColor(renderer, color, color, color, 255);
            SDL_Rect rect = {x*CELL_SIZE + 1, y*CELL_SIZE + 1, CELL_SIZE - 2, CELL_SIZE - 2};
            SDL_RenderFillRect(renderer, &rect);
        }
    }
}

void clear_canvas() {
    memset(pixels, 0, sizeof(pixels));
}

void update_pixels(int mx, int my, float intensity) {
    if (mx < 0 || mx >= WIDTH || my < 0 || my >= HEIGHT) return;

    int x = mx / CELL_SIZE;
    int y = my / CELL_SIZE;

    // Gaussian-like brush for smoother drawing
    for (int dy = -1; dy <= 1; dy++) {
        for (int dx = -1; dx <= 1; dx++) {
            int nx = x + dx;
            int ny = y + dy;
            if (nx >= 0 && nx < 28 && ny >= 0 && ny < 28) {
                float dist = sqrtf(dx*dx + dy*dy);
                float value = intensity * expf(-dist * dist / 2.0f);
                pixels[ny][nx] = fminf(1.0f, pixels[ny][nx] + value);
            }
        }
    }
}

void draw_confidence(Network *net) {
    // Draw background
    SDL_SetRenderDrawColor(renderer, 30, 30, 30, 255);
    SDL_Rect bg = {WIDTH, 0, 200, HEIGHT};
    SDL_RenderFillRect(renderer, &bg);
    
    Layer *output_layer = &net->layers[net->num_layers - 1];
    float *conf = output_layer->neurons;
    
    // Find the predicted digit
    int predicted = 0;
    float max_conf = conf[0];
    for (int i = 1; i < OUTPUT; i++) {
        if (conf[i] > max_conf) {
            max_conf = conf[i];
            predicted = i;
        }
    }
    
    for (int i = 0; i < OUTPUT; i++) {
        int bar_width = (int)(conf[i] * 150);
        int y_pos = i * 28 + 2;
        
        // Highlight predicted digit
        SDL_Color label_color = (i == predicted) ? (SDL_Color){255, 255, 100, 255} : text_color;
        
        // Draw label
        char label[3];
        snprintf(label, sizeof(label), "%d:", i);
        SDL_Surface* label_surface = TTF_RenderText_Solid(font, label, label_color);
        SDL_Texture* label_texture = SDL_CreateTextureFromSurface(renderer, label_surface);
        SDL_Rect label_rect = {
            WIDTH + 5,
            y_pos + 2,
            label_surface->w,
            label_surface->h
        };
        SDL_RenderCopy(renderer, label_texture, NULL, &label_rect);
        SDL_FreeSurface(label_surface);
        SDL_DestroyTexture(label_texture);
        
        // Draw bar background
        SDL_SetRenderDrawColor(renderer, 60, 60, 60, 255);
        SDL_Rect bg_bar = {WIDTH + 30, y_pos, 150, 20};
        SDL_RenderFillRect(renderer, &bg_bar);
        
        // Draw confidence bar
        if (i == predicted) {
            SDL_SetRenderDrawColor(renderer, 100, 200, 100, 255);
        } else {
            SDL_SetRenderDrawColor(renderer, 100, 100, 200, 255);
        }
        SDL_Rect bar = {WIDTH + 30, y_pos, bar_width, 20};
        SDL_RenderFillRect(renderer, &bar);
        
        // Draw percentage text
        char percentage[8];
        snprintf(percentage, sizeof(percentage), "%.1f%%", conf[i] * 100);
        SDL_Surface* text_surface = TTF_RenderText_Solid(font, percentage, text_color);
        SDL_Texture* text_texture = SDL_CreateTextureFromSurface(renderer, text_surface);
        
        SDL_Rect text_rect = {
            WIDTH + 35 + bar_width,
            y_pos + 2,
            text_surface->w,
            text_surface->h
        };
        
        // If bar is too small, put text at the beginning
        if (bar_width < 40) {
            text_rect.x = WIDTH + 35;
        }
        
        SDL_RenderCopy(renderer, text_texture, NULL, &text_rect);
        SDL_FreeSurface(text_surface);
        SDL_DestroyTexture(text_texture);
    }
    
    // Draw instructions
    SDL_SetRenderDrawColor(renderer, 200, 200, 200, 255);
    const char* instructions[] = {
        "Left click: Draw",
        "Right click: Erase", 
        "C: Clear canvas",
        "ESC: Quit"
    };
    
    int y_offset = HEIGHT - 100;
    for (int i = 0; i < 4; i++) {
        SDL_Surface* inst_surface = TTF_RenderText_Solid(font, instructions[i], text_color);
        SDL_Texture* inst_texture = SDL_CreateTextureFromSurface(renderer, inst_surface);
        SDL_Rect inst_rect = {
            WIDTH + 10,
            y_offset + i * 20,
            inst_surface->w,
            inst_surface->h
        };
        SDL_RenderCopy(renderer, inst_texture, NULL, &inst_rect);
        SDL_FreeSurface(inst_surface);
        SDL_DestroyTexture(inst_texture);
    }
}

void normalize_input(float *input) {
    // Center the drawing
    int min_x = 28, max_x = 0, min_y = 28, max_y = 0;
    
    for (int y = 0; y < 28; y++) {
        for (int x = 0; x < 28; x++) {
            if (pixels[y][x] > 0.1f) {
                if (x < min_x) min_x = x;
                if (x > max_x) max_x = x;
                if (y < min_y) min_y = y;
                if (y > max_y) max_y = y;
            }
        }
    }
    
    if (min_x <= max_x && min_y <= max_y) {
        int width = max_x - min_x + 1;
        int height = max_y - min_y + 1;
        int offset_x = (28 - width) / 2 - min_x;
        int offset_y = (28 - height) / 2 - min_y;
        
        float temp[28][28] = {0};
        for (int y = min_y; y <= max_y; y++) {
            for (int x = min_x; x <= max_x; x++) {
                int new_x = x + offset_x;
                int new_y = y + offset_y;
                if (new_x >= 0 && new_x < 28 && new_y >= 0 && new_y < 28) {
                    temp[new_y][new_x] = pixels[y][x];
                }
            }
        }
        
        memcpy(pixels, temp, sizeof(pixels));
    }
    
    // Copy to input array
    for (int i = 0; i < INPUT; i++) {
        input[i] = pixels[i/28][i%28];
    }
}

int main() {
    // Load neural network
    Network *net = load_network("weights.bin");
    if (!net) {
        printf("Failed to load neural network weights!\n");
        return 1;
    }
    
    // Initialize SDL
    if (SDL_Init(SDL_INIT_VIDEO) < 0) {
        printf("SDL init failed: %s\n", SDL_GetError());
        free_network(net);
        return 1;
    }
    
    if (TTF_Init() < 0) {
        printf("TTF init failed: %s\n", TTF_GetError());
        SDL_Quit();
        free_network(net);
        return 1;
    }
    
    // Try multiple font paths
    const char* font_paths[] = {
        "arial.ttf",
        "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/System/Library/Fonts/Helvetica.ttc",
        "C:\\Windows\\Fonts\\arial.ttf"
    };
    
    font = NULL;
    for (int i = 0; i < 5; i++) {
        font = TTF_OpenFont(font_paths[i], 14);
        if (font) break;
    }
    
    if (!font) {
        printf("Failed to load font. Please ensure a TrueType font is available.\n");
        TTF_Quit();
        SDL_Quit();
        free_network(net);
        return 1;
    }
    
    window = SDL_CreateWindow("MNIST Draw Test", 
                            SDL_WINDOWPOS_CENTERED,
                            SDL_WINDOWPOS_CENTERED, 
                            WIDTH + 200, HEIGHT, 
                            SDL_WINDOW_SHOWN);
    
    renderer = SDL_CreateRenderer(window, -1, 
                                SDL_RENDERER_ACCELERATED | SDL_RENDERER_PRESENTVSYNC);
    
    SDL_Event event;
    int running = 1;
    int mouse_down = 0;
    int erase_mode = 0;
    
    printf("\nInstructions:\n");
    printf("- Left click and drag to draw\n");
    printf("- Right click and drag to erase\n");
    printf("- Press 'C' to clear the canvas\n");
    printf("- Press 'ESC' to quit\n\n");
    
    while (running) {
        while (SDL_PollEvent(&event)) {
            switch (event.type) {
                case SDL_QUIT:
                    running = 0;
                    break;
                    
                case SDL_MOUSEBUTTONDOWN:
                    if (event.button.button == SDL_BUTTON_LEFT) {
                        mouse_down = 1;
                        erase_mode = 0;
                        update_pixels(event.button.x, event.button.y, 0.8f);
                    } else if (event.button.button == SDL_BUTTON_RIGHT) {
                        mouse_down = 1;
                        erase_mode = 1;
                        update_pixels(event.button.x, event.button.y, -0.5f);
                    }
                    break;
                    
                case SDL_MOUSEBUTTONUP:
                    mouse_down = 0;
                    break;
                    
                case SDL_MOUSEMOTION:
                    if (mouse_down) {
                        float intensity = erase_mode ? -0.5f : 0.8f;
                        update_pixels(event.motion.x, event.motion.y, intensity);
                    }
                    break;
                    
                case SDL_KEYDOWN:
                    if (event.key.keysym.sym == SDLK_c) {
                        clear_canvas();
                    } else if (event.key.keysym.sym == SDLK_ESCAPE) {
                        running = 0;
                    }
                    break;
            }
        }
        
        // Prepare network input
        float input[INPUT];
        normalize_input(input);
        
        // Run prediction
        forward_pass(net, input);
        
        // Render
        SDL_SetRenderDrawColor(renderer, 20, 20, 20, 255);
        SDL_RenderClear(renderer);
        
        draw_pixels();
        draw_grid();
        draw_confidence(net);
        
        SDL_RenderPresent(renderer);
    }
    
    // Cleanup
    free_network(net);
    TTF_CloseFont(font);
    TTF_Quit();
    SDL_DestroyRenderer(renderer);
    SDL_DestroyWindow(window);
    SDL_Quit();
    
    return 0;
}
