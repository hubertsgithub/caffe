import numpy as np
import os
import sys
import random

import globals
import html
import intrinsic

sys.path.append('data')
sys.path.append('data/iiw-dataset')
import whdr
import common

from celery import Celery

globals.init()

SAVEROOTDIR = 'experiments/mitintrinsic/allresults'

# The following objects were used in the evaluation. For the learning algorithms
# (not included here), we used two-fold cross-validation with the following
# randomly chosen split.
SET1MIT = ['box', 'cup1', 'cup2', 'dinosaur', 'panther', 'squirrel', 'sun', 'teabag2']
SET2MIT = ['deer', 'frog1', 'frog2', 'paper1', 'paper2', 'raccoon', 'teabag1', 'turtle']

SETINDOOR = map(lambda n: str(n), range(1, 25))

random.seed(10)
with open('data/iiw-dataset/denseimages.txt') as f:
    SETIIWDENSE = random.sample([s.strip() for s in f.readlines()], 10)

if globals.DATASETCHOICE == 0:
    ALL_TAGS = SET1MIT + SET2MIT
    ERRORMETRIC = 0  # LMSE
elif globals.DATASETCHOICE == 1:
    ALL_TAGS = SETINDOOR
    ERRORMETRIC = 0  # LMSE
elif globals.DATASETCHOICE == 2:
    ALL_TAGS = SETIIWDENSE
    ERRORMETRIC = 1  # WHDR
else:
    raise ValueError('Unknown dataset choice: {0}'.format(globals.DATASETCHOICE))

# The following four objects weren't used in the evaluation because they have
# slight problems, but you may still find them useful.
EXTRA_TAGS = ['apple', 'pear', 'phone', 'potato']

# Use L1 to compute the final results. (For efficiency, the parameters are still
# tuned using least squares.)
USE_L1 = False

# Output of the algorithms will be saved here
if USE_L1:
    RESULTS_DIR = os.path.join(SAVEROOTDIR, 'results_L1')
else:
    RESULTS_DIR = os.path.join(SAVEROOTDIR, 'results')

estimators = [
                ('Baseline (BAS)', intrinsic.BaselineEstimator),
                #('Grayscale Retinex with CNN predicted threshold images using RGB images', intrinsic.GrayscaleRetinexWithThresholdImageRGBEstimator),
                #('Grayscale Retinex with CNN predicted threshold images using chromaticity + grayscale image, small network 3 conv layers', intrinsic.GrayscaleRetinexWithThresholdImageChromSmallNetEstimator),
                #('Grayscale Retinex with CNN predicted threshold images using chromaticity + grayscale image, big network 4 conv layers', intrinsic.GrayscaleRetinexWithThresholdImageChromBigNetEstimator),
                #('Grayscale Retinex with CNN predicted threshold images using chromaticity + grayscale image, big network 4 conv layers, concatenated conv1+3 output', intrinsic.GrayscaleRetinexWithThresholdImageChromBigNetConcatEstimator),
                #('Grayscale Retinex with CNN predicted threshold images using chromaticity + grayscale image, big network 4 conv layers, concatenated conv1+3 output + maxpool between conv1-2 and 2-3', intrinsic.GrayscaleRetinexWithThresholdImageChromBigNetConcatMaxpoolEstimator),
                #('Grayscale Retinex with ground truth threshold images', intrinsic.GrayscaleRetinexWithThresholdImageGroundTruthEstimator),
                ('Zhao2012', intrinsic.Zhao2012Estimator),
                ('Zhao2012 with ground truth reflectance groups', intrinsic.Zhao2012GroundTruthGroupsEstimator),
                ('Grayscale Retinex (GR-RET)', intrinsic.GrayscaleRetinexEstimator),
                ('Color Retinex (COL-RET)', intrinsic.ColorRetinexEstimator),
                #("Weiss's Algorithm (W)", intrinsic.WeissEstimator),
                #('Weiss + Retinex (W+RET)', intrinsic.WeissRetinexEstimator),
                ]

app = Celery('comparison', broker='amqp://guest@localhost//')

def print_dot(i, num):
    NEWLINE_EVERY = 50
    sys.stdout.write('.')
    if (i+1) % NEWLINE_EVERY == 0:
        sys.stdout.write('  [%d/%d]' % (i+1, num))
    if (i+1) % NEWLINE_EVERY == 0 or i+1 == num:
        sys.stdout.write('\n')
    sys.stdout.flush()


def save_estimates(gen, image, est_shading, est_refl, mask):
    """Outputs the estimated shading and reflectance images to an HTML
    file. Does nothing if Python Imaging Library is not installed."""
    image = image / np.max(image)
    est_shading = est_shading / np.max(est_shading)
    est_refl = est_refl / np.max(est_refl)
    est_refl = common.compute_color_reflectance(est_refl, image)

    # gamma correct
    image = np.where(mask[:, :, np.newaxis], common.rgb_to_srgb(image), 1.)
    est_shading = np.where(mask, common.rgb_to_srgb(est_shading), 1.)
    est_refl = np.where(mask[:, :, np.newaxis], common.rgb_to_srgb(est_refl), 1.)

    # create RGB shading image from grayscale
    est_shading = est_shading[:, :, np.newaxis].repeat(3, axis=2)

    output = np.concatenate([image, est_shading, est_refl], axis=1)
    gen.image(output)


def run_experiment():
    """Script for running the algorithmic comparisons from the paper

        Roger Grosse, Micah Johnson, Edward Adelson, and William Freeman,
          Ground truth dataset and baseline evaluations for intrinsic
          image algorithms.

    Evaluates each of the algorithms on the MIT Intrinsic Images dataset
    with hold-one-out cross-validation.

    For each algorithm, it precomputes the error scores for all objects with
    all parameter settings. Then, for each object, it chooses the parameters
    which achieved the smallest average error on the other objects. The
    results are all output to the HTML file results/index.html."""

    assert os.path.isdir(RESULTS_DIR), '%s: directory does not exist' % RESULTS_DIR

    tags = ALL_TAGS
    ntags = len(tags)

    gen = html.Generator('Intrinsic image results', RESULTS_DIR)

    results = np.zeros((len(estimators), ntags))
    for e, (name, EstimatorClass) in enumerate(estimators):
        print 'Evaluating %s' % name
        sys.stdout.flush()
        gen.heading(name)

        choices = EstimatorClass.param_choices()
        nchoices = len(choices)

        # Try all parameters on all the objects
        scores = np.zeros((ntags, nchoices))
        count = 0
        for i, tag in enumerate(tags):
            # Estimators know what input they expect (grayscale image, color image, etc.)
            inp = EstimatorClass.get_input(tag)

            if ERRORMETRIC == 0:
                true_shading = intrinsic.load_object(tag, 'shading')
                true_refl = intrinsic.load_object(tag, 'reflectance')
                true_refl = np.mean(true_refl, axis=2)
                mask = intrinsic.load_object(tag, 'mask')
            elif ERRORMETRIC == 1:
                judgements = intrinsic.load_object(tag, 'judgements')
            else:
                raise ValueError('Unknown error metric choice: {0}'.format(ERRORMETRIC))

            print 'Estimating shading and reflectance for ' + tag

            for j, params in enumerate(choices):
                estimator = EstimatorClass(**params)
                est_shading, est_refl = estimator.estimate_shading_refl(*inp)

                if ERRORMETRIC == 0:
                    scores[i, j] = intrinsic.score_image(true_shading, true_refl, est_shading, est_refl, mask)
                elif ERRORMETRIC == 1:
                    scores[i, j] = whdr.compute_whdr(est_refl, judgements)
                else:
                    raise ValueError('Unknown error metric choice: {0}'.format(ERRORMETRIC))

                print 'Params: {0}'.format(params)
                print 'Score: {0}'.format(scores[i, j])
                print_dot(count, ntags * nchoices)
                count += 1

        # Hold-one-out cross-validation
        print '  Final scores:'
        sys.stdout.flush()
        for i, tag in enumerate(tags):
            inp = EstimatorClass.get_input(tag)
            inp = inp + (USE_L1,)

            image = intrinsic.load_object(tag, 'diffuse')
            mask = intrinsic.load_object(tag, 'mask')

            if ERRORMETRIC == 0:
                true_shading = intrinsic.load_object(tag, 'shading')
                true_refl = intrinsic.load_object(tag, 'reflectance')
                true_refl = np.mean(true_refl, axis=2)
            elif ERRORMETRIC == 1:
                judgements = intrinsic.load_object(tag, 'judgements')
            else:
                raise ValueError('Unknown error metric choice: {0}'.format(ERRORMETRIC))

            other_inds = range(i) + range(i+1, ntags)
            total_scores = np.sum(scores[other_inds, :], axis=0)
            best_choice = np.argmin(total_scores)
            params = choices[best_choice]
            estimator = EstimatorClass(**params)
            est_shading, est_refl = estimator.estimate_shading_refl(*inp)

            if ERRORMETRIC == 0:
                score = intrinsic.score_image(true_shading, true_refl, est_shading, est_refl, mask)
            elif ERRORMETRIC == 1:
                score = whdr.compute_whdr(est_refl, judgements)
            else:
                raise ValueError('Unknown error metric choice: {0}'.format(ERRORMETRIC))

            gen.text('%s: %1.3f' % (tag, score))

            save_estimates(gen, image, est_shading, est_refl, mask)

            print '    %s: %1.3f' % (tag, score)

            results[e, i] = score
        print '    average: %1.3f' % np.mean(results[e, :])

        gen.divider()

    gen.heading('Mean error')
    for e, (name, EstimatorClass) in enumerate(estimators):
        avg = np.mean(results[e, :])
        gen.text('%s: %1.3f' % (name, avg))


def run_parallel_experiment():
    """Script for running the algorithmic comparisons from the paper

        Roger Grosse, Micah Johnson, Edward Adelson, and William Freeman,
          Ground truth dataset and baseline evaluations for intrinsic
          image algorithms.

    Evaluates each of the algorithms on the MIT Intrinsic Images dataset
    with hold-one-out cross-validation.

    For each algorithm, it precomputes the error scores for all objects with
    all parameter settings. Then, for each object, it chooses the parameters
    which achieved the smallest average error on the other objects. The
    results are all output to the HTML file results/index.html."""

    assert os.path.isdir(RESULTS_DIR), '%s: directory does not exist' % RESULTS_DIR

    tags = ALL_TAGS
    ntags = len(tags)

    results = np.zeros((len(estimators), ntags))
    # Generate HTML using the results
    gen = html.Generator('Intrinsic image results', RESULTS_DIR)

    for e, (name, EstimatorClass) in enumerate(estimators):
        print 'Evaluating (starting jobs) %s' % name
        sys.stdout.flush()

        choices = EstimatorClass.param_choices()
        nchoices = len(choices)

        # Try all parameters on all the objects
        scores = np.zeros((ntags, nchoices))

        for i, tag in enumerate(tags):
            for j, params in enumerate(choices):
                computeScoreJob(name, EstimatorClass, params, tag, i, j)

        # Collect results from files
        for i, tag in enumerate(tags):
            for j, params in enumerate(choices):
                with open(os.path.join(RESULTS_DIR, '{0}_{1}.txt'.format(i, j)), 'r') as f:
                    scores[i, j] = float(f.read())

        gen.heading(name)

        # Hold-one-out cross-validation
        print '  Final scores:'
        sys.stdout.flush()
        for i, tag in enumerate(tags):
            inp = EstimatorClass.get_input(tag)
            inp = inp + (USE_L1,)

            image = intrinsic.load_object(tag, 'diffuse')
            mask = intrinsic.load_object(tag, 'mask')

            if ERRORMETRIC == 0:
                true_shading = intrinsic.load_object(tag, 'shading')
                true_refl = intrinsic.load_object(tag, 'reflectance')
                true_refl = np.mean(true_refl, axis=2)
            elif ERRORMETRIC == 1:
                judgements = intrinsic.load_object(tag, 'judgements')
            else:
                raise ValueError('Unknown error metric choice: {0}'.format(ERRORMETRIC))

            other_inds = range(i) + range(i+1, ntags)
            total_scores = np.sum(scores[other_inds, :], axis=0)
            best_choice = np.argmin(total_scores)
            params = choices[best_choice]
            estimator = EstimatorClass(**params)
            est_shading, est_refl = estimator.estimate_shading_refl(*inp)

            if ERRORMETRIC == 0:
                score = intrinsic.score_image(true_shading, true_refl, est_shading, est_refl, mask)
            elif ERRORMETRIC == 1:
                score = whdr.compute_whdr(est_refl, judgements)
            else:
                raise ValueError('Unknown error metric choice: {0}'.format(ERRORMETRIC))

            gen.text('%s: %1.3f' % (tag, score))

            save_estimates(gen, image, est_shading, est_refl, mask)

            print '    %s: %1.3f' % (tag, score)

            results[e, i] = score
        print '    average: %1.3f' % np.mean(results[e, :])

        gen.divider()

    gen.heading('Mean error')
    for e, (name, EstimatorClass) in enumerate(estimators):
        avg = np.mean(results[e, :])
        gen.text('%s: %1.3f' % (name, avg))


@app.task
def computeScoreJob(name, EstimatorClass, params, tag, i, j):
    """
    Input paramsDict:
        {'estimator' : (name, EstimatorClass)}
        {'params' : params with which the estimation will be called}
        {'tag' : the estimated object's (image) tag}
        {'score_coords' : the coordinates in the score matrix (i, j)}
    """

    # Estimators know what input they expect (grayscale image, color image, etc.)
    inp = EstimatorClass.get_input(tag)

    if ERRORMETRIC == 0:
        true_shading = intrinsic.load_object(tag, 'shading')
        true_refl = intrinsic.load_object(tag, 'reflectance')
        true_refl = np.mean(true_refl, axis=2)
        mask = intrinsic.load_object(tag, 'mask')
    elif ERRORMETRIC == 1:
        judgements = intrinsic.load_object(tag, 'judgements')
    else:
        raise ValueError('Unknown error metric choice: {0}'.format(ERRORMETRIC))

    estimator = EstimatorClass(**params)
    est_shading, est_refl = estimator.estimate_shading_refl(*inp)

    if ERRORMETRIC == 0:
        score = intrinsic.score_image(true_shading, true_refl, est_shading, est_refl, mask)
    elif ERRORMETRIC == 1:
        score = whdr.compute_whdr(est_refl, judgements)
    else:
        raise ValueError('Unknown error metric choice: {0}'.format(ERRORMETRIC))

    # Write results to file
    with open(os.path.join(RESULTS_DIR, '{0}_{1}.txt'.format(i, j)), 'w') as f:
        f.write(str(score))


if __name__ == '__main__':
    run_parallel_experiment()
