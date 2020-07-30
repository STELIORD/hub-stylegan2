import os
import gradio as gr
import wget
import pretrained_networks
import numpy as np
import dnnlib
import dnnlib.tflib as tflib

if not os.path.exists("network-snapshot-006746.pkl"):
    wget.download("https://archive.org/download/wikiart-stylegan2-conditional-model/network-snapshot-006746.pkl", "network-snapshot-006746.pkl")


network_pkl = 'network-snapshot-006746.pkl'
_G, _D, Gs = pretrained_networks.load_networks(network_pkl)

Gs_syn_kwargs = dnnlib.EasyDict()
batch_size = 1
Gs_syn_kwargs.output_transform = dict(func=tflib.convert_images_to_uint8, nchw_to_nhwc=True)
Gs_syn_kwargs.randomize_noise = True
Gs_syn_kwargs.minibatch_size = batch_size


artists_zipped =[('Unknown Artist', 0), ('Boris Kustodiev', 1), ('Camille Pissarro', 2), ('Childe Hassam', 3), ('Claude Monet', 4), ('Edgar Degas', 5), ('Eugene Boudin', 6), ('Gustave Dore', 7), ('Ilya Repin', 8), ('Ivan Aivazovsky', 9), ('Ivan Shishkin', 10), ('John Singer Sargent', 11), ('Marc Chagall', 12), ('Martiros Saryan', 13), ('Nicholas Roerich', 14), ('Pablo Picasso', 15), ('Paul Cezanne', 16), ('Pierre Auguste Renoir', 17), ('Pyotr Konchalovsky', 18), ('Raphael Kirchner', 19), ('Rembrandt', 20), ('Salvador Dali', 21), ('Vincent Van Gogh', 22), ('Hieronymus Bosch', 23), ('Leonardo Da Vinci', 24), ('Albrecht Durer', 25), ('Edouard Cortes', 26), ('Sam Francis', 27), ('Juan Gris', 28), ('Lucas Cranach The Elder', 29), ('Paul Gauguin', 30), ('Konstantin Makovsky', 31), ('Egon Schiele', 32), ('Thomas Eakins', 33), ('Gustave Moreau', 34), ('Francisco Goya', 35), ('Edvard Munch', 36), ('Henri Matisse', 37), ('Fra Angelico', 38), ('Maxime Maufra', 39), ('Jan Matejko', 40), ('Mstislav Dobuzhinsky', 41), ('Alfred Sisley', 42), ('Mary Cassatt', 43), ('Gustave Loiseau', 44), ('Fernando Botero', 45), ('Zinaida Serebriakova', 46), ('Georges Seurat', 47), ('Isaac Levitan', 48), ('Joaquã­n Sorolla', 49), ('Jacek Malczewski', 50), ('Berthe Morisot', 51), ('Andy Warhol', 52), ('Arkhip Kuindzhi', 53), ('Niko Pirosmani', 54), ('James Tissot', 55), ('Vasily Polenov', 56), ('Valentin Serov', 57), ('Pietro Perugino', 58), ('Pierre Bonnard', 59), ('Ferdinand Hodler', 60), ('Bartolome Esteban Murillo', 61), ('Giovanni Boldini', 62), ('Henri Martin', 63), ('Gustav Klimt', 64), ('Vasily Perov', 65), ('Odilon Redon', 66), ('Tintoretto', 67), ('Gene Davis', 68), ('Raphael', 69), ('John Henry Twachtman', 70), ('Henri De Toulouse Lautrec', 71), ('Antoine Blanchard', 72), ('David Burliuk', 73), ('Camille Corot', 74), ('Konstantin Korovin', 75), ('Ivan Bilibin', 76), ('Titian', 77), ('Maurice Prendergast', 78), ('Edouard Manet', 79), ('Peter Paul Rubens', 80), ('Aubrey Beardsley', 81), ('Paolo Veronese', 82), ('Joshua Reynolds', 83), ('Kuzma Petrov Vodkin', 84), ('Gustave Caillebotte', 85), ('Lucian Freud', 86), ('Michelangelo', 87), ('Dante Gabriel Rossetti', 88), ('Felix Vallotton', 89), ('Nikolay Bogdanov Belsky', 90), ('Georges Braque', 91), ('Vasily Surikov', 92), ('Fernand Leger', 93), ('Konstantin Somov', 94), ('Katsushika Hokusai', 95), ('Sir Lawrence Alma Tadema', 96), ('Vasily Vereshchagin', 97), ('Ernst Ludwig Kirchner', 98), ('Mikhail Vrubel', 99), ('Orest Kiprensky', 100), ('William Merritt Chase', 101), ('Aleksey Savrasov', 102), ('Hans Memling', 103), ('Amedeo Modigliani', 104), ('Ivan Kramskoy', 105), ('Utagawa Kuniyoshi', 106), ('Gustave Courbet', 107), ('William Turner', 108), ('Theo Van Rysselberghe', 109), ('Joseph Wright', 110), ('Edward Burne Jones', 111), ('Koloman Moser', 112), ('Viktor Vasnetsov', 113), ('Anthony Van Dyck', 114), ('Raoul Dufy', 115), ('Frans Hals', 116), ('Hans Holbein The Younger', 117), ('Ilya Mashkov', 118), ('Henri Fantin Latour', 119), ('M.C. Escher', 120), ('El Greco', 121), ('Mikalojus Ciurlionis', 122), ('James Mcneill Whistler', 123), ('Karl Bryullov', 124), ('Jacob Jordaens', 125), ('Thomas Gainsborough', 126), ('Eugene Delacroix', 127), ('Canaletto', 128)]
artists_mapping = {artist:value for artist, value in artists_zipped}
artists = gr.inputs.Dropdown(
    choices = list(artists_mapping.keys()),
    label="Artist"
)

genres_zipped =[('Abstract Painting', 129), ('Cityscape', 130), ('Genre Painting', 131), ('Illustration', 132), ('Landscape', 133), ('Nude Painting', 134), ('Portrait', 135), ('Religious Painting', 136), ('Sketch And Study', 137), ('Still Life', 138), ('Unknown Genre', 139)]
genres_mapping = {genre:value for genre, value in genres_zipped}


genres = gr.inputs.Dropdown(
    choices = list(genres_mapping.keys()),
    label='Genre'
)

styles_zipped = [('Abstract Expressionism', 140), ('Action Painting', 141), ('Analytical Cubism', 142), ('Art Nouveau', 143), ('Baroque', 144), ('Color Field Painting', 145), ('Contemporary Realism', 146), ('Cubism', 147), ('Early Renaissance', 148), ('Expressionism', 149), ('Fauvism', 150), ('High Renaissance', 151), ('Impressionism', 152), ('Mannerism Late Renaissance', 153), ('Minimalism', 154), ('Naive Art Primitivism', 155), ('New Realism', 156), ('Northern Renaissance', 157), ('Pointillism', 158), ('Pop Art', 159), ('Post Impressionism', 160), ('Realism', 161), ('Rococo', 162), ('Romanticism', 163), ('Symbolism', 164), ('Synthetic Cubism', 165), ('Ukiyo-e', 166)]
styles_mapping = {style:value for style, value in styles_zipped}


styles = gr.inputs.Dropdown(
    choices = list(styles_mapping.keys()),
    label='Style'
)

seed = gr.inputs.Slider(minimum=0, maximum=99999, default=0, label="Seed")
scale = gr.inputs.Slider(minimum=0, maximum=5, default=1, label='Scale')
truncation = gr.inputs.Slider(minimum=0.5, maximum=2, default=1, label='Truncation')


def generate_image(artist, genre, style, seed, scale, truncation):
    artist = artists_mapping[artist]
    genre = genres_mapping[genre]
    style = styles_mapping[style]

    batch_size = 1
    l1 = np.zeros((1,167))
    l1[0][artist] = 1
    l1[0][genre] = 1
    l1[0][style] = 1
    all_seeds = [seed] * batch_size
    all_z = np.stack([np.random.RandomState(seed).randn(*Gs.input_shape[1:]) for seed in all_seeds]) # [minibatch, component]
    all_w = Gs.components.mapping.run(scale*all_z, np.tile(l1, (batch_size, 1))) # [minibatch, layer, component]
    if truncation != 1:
        w_avg = Gs.get_var('dlatent_avg')
        all_w = w_avg + (all_w - w_avg) * truncation # [minibatch, layer, component]
    all_images = Gs.components.synthesis.run(all_w, **Gs_syn_kwargs)
    return np.squeeze(all_images, axis=0)

title="Painting Generator (StyleGAN)"
description="This GAN model, trained on WikiArt images, generates a never-seen-before painting based on an artist, style, and genre!"
examples=[
["Leonardo Da Vinci","Portrait","High Renaissance",0,1,0.5],
["Pablo Picasso", "Landscape", "Cubism",0,0.7,1],
["M.C. Escher", "Abstract Painting", "Abstract Expressionism",0,1,1]
]
thumbnail="https://github.com/gradio-app/hub-stylegan2/raw/master/screenshot" \
          ".png?raw=true"
gr.Interface(generate_image, [artists, genres, styles, seed, scale, truncation], gr.outputs.Image(), capture_session=True, thumbnail=thumbnail,
             title=title, description=description, examples=examples).launch()
