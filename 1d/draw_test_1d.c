#include <SDL2/SDL_render.h>
#include <SDL2/SDL_shape.h>
#include <stdlib.h>
#include <stdio.h>
#include <SDL2/SDL.h>
#include <SDL2/SDL_ttf.h>
#include <math.h>

#define WIDTH 280
#define HEIGHT 280
#define CELL_SIZE (WIDTH / 28)

typedef struct {
    int size;
    int pre_size;
    float *neurons;
    float *biases;
    float *weights;
} Layer;

SDL_Window *window = NULL;
SDL_Renderer *renderer = NULL;
TTF_Font *font = NULL;
SDL_Color text_color = {255, 255, 255, 255};
float pixels[28][28] = {0};

void init_layer(Layer *l, int pre_size, int size) {
    l->size = size;
    l->pre_size = pre_size;
    l->neurons = (float*)malloc(size * sizeof(float));
    l->biases = (float*)malloc(size * sizeof(float));
    l->weights = (float*)malloc(size * pre_size * sizeof(float));
}

void init(Layer l[], int sizes[], int lay_num) {
    for(int i = 0; i < lay_num; i++) {
        init_layer(&l[i], sizes[i], sizes[i+1]);
    }
}

void free_layer(Layer *l) {
    if(!l) return;

    free(l->weights);
    free(l->neurons);
    free(l->biases);
}

void feedforward(float input[], int size, Layer *l, int activate) {
    if(!input || !l->neurons || !l->biases) {
        printf("Invalid layer in feedforward.\n");
        return;
    }

    for(int i = 0; i < l->size; i++) {
        l->neurons[i] = l->biases[i];
        for(int j = 0; j < size; j++) {
            l->neurons[i] += input[j] * l->weights[i * size + j];
        }
        if(activate) {
            l->neurons[i] = fmaxf(0.0f, l->neurons[i]);
        }
    }
}

void softmax(float output[], int length) {
    float max = output[0];
    for(int i = 1; i < length; i++)
        if(output[i] > max) max = output[i];

    float sum = 0.0;
    for(int i = 0; i < length; i++) {
        output[i] = exp(output[i] - max);
        sum += output[i];
    }

    for(int i = 0; i < length; i++)
        output[i] /= sum;
}

Layer* load_model(const char* name, int* lay_num, int** lay_sizes_ptr) {
    FILE *f = fopen(name, "rb");
    if(!f) {
        printf("Could NOT open \"%s\" file.\n", name);
        return NULL;
    }

    // Read layer count
    if (fread(lay_num, sizeof(int), 1, f) != 1) {
        printf("Error reading layer count from %s\n", name);
        fclose(f);
        return NULL;
    }

    // Allocate and read layer sizes
    *lay_sizes_ptr = (int*)malloc((*lay_num + 1) * sizeof(int));
    if (fread(*lay_sizes_ptr, sizeof(int), *lay_num + 1, f) != (*lay_num + 1)) {
        printf("Error reading layer sizes from %s\n", name);
        fclose(f);
        free(*lay_sizes_ptr);
        return NULL;
    }

    // Allocate and initialize layers
    Layer *l = (Layer*)malloc(*lay_num * sizeof(Layer));
    init(l, *lay_sizes_ptr, *lay_num); // Uses the local init()

    // Read weights and biases
    for(int layer = 0; layer < *lay_num; layer++) {
        fread(l[layer].biases, sizeof(float), l[layer].size, f);
        fread(l[layer].weights, sizeof(float), l[layer].pre_size * l[layer].size, f);
    }

    fclose(f);
    printf("Model loaded from %s\n", name);
    return l;
}

// Drawing functions
void draw_grid() {
    SDL_SetRenderDrawColor(renderer, 200, 200, 200, 255);
    for(int i = 0; i <= 28; i++) {
        SDL_RenderDrawLine(renderer, i*CELL_SIZE, 0, i*CELL_SIZE, HEIGHT);
        SDL_RenderDrawLine(renderer, 0, i*CELL_SIZE, WIDTH, i*CELL_SIZE);
    }
}

void draw_pixels() {
    for(int y = 0; y < 28; y++) {
        for(int x = 0; x < 28; x++) {
            int color = (int)(pixels[y][x] * 255);
            SDL_SetRenderDrawColor(renderer, color, color, color, 255);
            SDL_Rect rect = {x*CELL_SIZE, y*CELL_SIZE, CELL_SIZE, CELL_SIZE};
            SDL_RenderFillRect(renderer, &rect);
        }
    }
}

void clear_canvas() {
    memset(pixels, 0, sizeof(pixels));
}

void update_pixels(int mx, int my) {
    if(mx < 0 || mx >= WIDTH || my < 0 || my >= HEIGHT) return;

    int x = mx / CELL_SIZE;
    int y = my / CELL_SIZE;

    // Paint surrounding cells for thicker drawing
    for(int dy = -1; dy <= 1; dy++) {
        for(int dx = -1; dx <= 1; dx++) {
            int nx = x + dx;
            int ny = y + dy;
            if(nx >= 0 && nx < 28 && ny >= 0 && ny < 28) {
                pixels[ny][nx] = fminf(1.0f, pixels[ny][nx] + 0.3f);
            }
        }
    }
}

void draw_confidence(float* conf) {
    // Draw background
    SDL_SetRenderDrawColor(renderer, 40, 40, 40, 255);
    SDL_Rect bg = {WIDTH, 0, 120, HEIGHT};
    SDL_RenderFillRect(renderer, &bg);

    for(int i = 0; i < 10; i++) {
        int bar_width = (int)(conf[i] * 100);

        // Draw label
        char label[2];
        snprintf(label, sizeof(label), "%d", i);
        SDL_Surface* label_surface = TTF_RenderText_Solid(font, label, text_color);
        SDL_Texture* label_texture = SDL_CreateTextureFromSurface(renderer, label_surface);
        SDL_Rect label_rect = {
            WIDTH + 5,
            i * 28 + 5,
            label_surface->w,
            label_surface->h
        };
        SDL_RenderCopy(renderer, label_texture, NULL, &label_rect);
        SDL_FreeSurface(label_surface);
        SDL_DestroyTexture(label_texture);

        // Draw bar background
        SDL_SetRenderDrawColor(renderer, 60, 60, 60, 255);
        SDL_Rect bg_bar = {WIDTH + 25, i*28 + 2, 100, 20};
        SDL_RenderFillRect(renderer, &bg_bar);

        // Draw confidence bar
        SDL_SetRenderDrawColor(renderer, 0, 200, 0, 255);
        SDL_Rect bar = {WIDTH + 25, i*28 + 2, bar_width, 20};
        SDL_RenderFillRect(renderer, &bar);

        // Draw percentage text
        char percentage[16];
        snprintf(percentage, sizeof(percentage), "%.1f%%", conf[i] * 100);
        SDL_Surface* text_surface = TTF_RenderText_Solid(font, percentage, text_color);
        SDL_Texture* text_texture = SDL_CreateTextureFromSurface(renderer, text_surface);

        SDL_Rect text_rect = {
            WIDTH + 25 + bar_width + 5,
            i * 28 + 2,
            text_surface->w,
            text_surface->h
        };
        // If bar is too small, put text inside
        if(bar_width < 35) {
            text_rect.x = WIDTH + 25;
            SDL_SetRenderDrawColor(renderer, 0, 0, 0, 255);
        }

        SDL_RenderCopy(renderer, text_texture, NULL, &text_rect);
        SDL_FreeSurface(text_surface);
        SDL_DestroyTexture(text_texture);
    }
}

void run(Layer *l, int lay_num, int lay_sizes[]) {
    // Initialize SDL
    if(SDL_Init(SDL_INIT_VIDEO) < 0) {
        printf("SDL init failed: %s\n", SDL_GetError());
        exit(1);
    }
    TTF_Init();

    font = TTF_OpenFont("../files/arial.ttf", 14);
    if(!font) {
        printf("Failed to load font: %s\n", TTF_GetError());
        exit(1);
    }

    window = SDL_CreateWindow("MNIST Draw", SDL_WINDOWPOS_CENTERED,
                            SDL_WINDOWPOS_CENTERED, WIDTH+120, HEIGHT, 0);
    renderer = SDL_CreateRenderer(window, -1, SDL_RENDERER_ACCELERATED);

    SDL_Event event;
    int running = 1;
    int mouse_down = 0;

    while(running) {
        while(SDL_PollEvent(&event)) {
            switch(event.type) {
                case SDL_QUIT:
                    running = 0;
                    break;
                case SDL_MOUSEBUTTONDOWN:
                    mouse_down = 1;
                    update_pixels(event.button.x, event.button.y);
                    break;
                case SDL_MOUSEBUTTONUP:
                    mouse_down = 0;
                    break;
                case SDL_MOUSEMOTION:
                    if(mouse_down)
                        update_pixels(event.motion.x, event.motion.y);
                    break;
                case SDL_KEYDOWN:
                    if(event.key.keysym.sym == SDLK_c)
                        clear_canvas();
                    break;
            }
        }

        // Prepare network input
        float input[lay_sizes[0]];
        for(int i = 0; i < lay_sizes[0]; i++) {
            input[i] = pixels[i/28][i%28];
        }

        feedforward(input, lay_sizes[0], &l[0], 1);
        for(int lay = 0; lay < lay_num-2; lay++) {
            feedforward(l[lay].neurons, lay_sizes[lay+1], &l[lay+1], 1);
        }
        feedforward(l[lay_num-2].neurons, lay_sizes[lay_num-1], &l[lay_num-1], 0);
        softmax(l[lay_num-1].neurons, lay_sizes[lay_num]);
        // Run prediction

        // Render
        SDL_SetRenderDrawColor(renderer, 255, 255, 255, 255);
        SDL_RenderClear(renderer);

        draw_pixels();
        draw_grid();
        draw_confidence(l[lay_num-1].neurons);

        SDL_RenderPresent(renderer);
    }

    for(int i = 0; i < lay_num; i++) {
        free_layer(&l[i]);
    }
    TTF_CloseFont(font);
    TTF_Quit();
    SDL_DestroyRenderer(renderer);
    SDL_DestroyWindow(window);
    SDL_Quit();
}

int main() {
    int lay_num;
    int *lay_sizes;
    Layer *l = load_model("weights.bin", &lay_num, &lay_sizes);
    if(!l) {
        return 1;
    }

    run(l, lay_num, lay_sizes);

    free(l);
    free(lay_sizes);
    return 0;
}
