import os
import sys

import numpy as np
import scipy as sp
import redis
import cPickle as pickle
import datetime

from lib.intrinsic import html, intrinsic
from lib.intrinsic import html, intrinsic, tasks, resulthandler
from lib.utils.data import common, whdr
from lib.utils.misc import packer


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


def run_experiment(DATASETCHOICE, ALL_TAGS, ERRORMETRIC, USE_L1, RESULTS_DIR, ESTIMATORS):
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

    results = np.zeros((len(ESTIMATORS), ntags))
    for e, (name, EstimatorClass) in enumerate(ESTIMATORS):
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
            inp = EstimatorClass.get_input(tag, DATASETCHOICE)

            if ERRORMETRIC == 0:
                true_shading = intrinsic.load_object(tag, 'shading', DATASETCHOICE)
                true_refl = intrinsic.load_object(tag, 'reflectance', DATASETCHOICE)
                true_refl = np.mean(true_refl, axis=2)
                mask = intrinsic.load_object(tag, 'mask', DATASETCHOICE)
            elif ERRORMETRIC == 1:
                judgements = intrinsic.load_object(tag, 'judgements', DATASETCHOICE)
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
            inp = EstimatorClass.get_input(tag, DATASETCHOICE)
            inp = inp + (USE_L1,)

            image = intrinsic.load_object(tag, 'diffuse', DATASETCHOICE)
            mask = intrinsic.load_object(tag, 'mask', DATASETCHOICE)

            if ERRORMETRIC == 0:
                true_shading = intrinsic.load_object(tag, 'shading', DATASETCHOICE)
                true_refl = intrinsic.load_object(tag, 'reflectance', DATASETCHOICE)
                true_refl = np.mean(true_refl, axis=2)
            elif ERRORMETRIC == 1:
                judgements = intrinsic.load_object(tag, 'judgements', DATASETCHOICE)
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
    for e, (name, EstimatorClass) in enumerate(ESTIMATORS):
        avg = np.mean(results[e, :])
        gen.text('%s: %1.3f' % (name, avg))


def dispatch_comparison_experiment(DATASETCHOICE, ALL_TAGS, ERRORMETRIC, USE_L1, RESULTS_DIR, ESTIMATORS):
    tags = ALL_TAGS
    all_processed = resulthandler.get_all_processed()

    for e, (name, EstimatorClass) in enumerate(ESTIMATORS):
        print 'Evaluating (starting jobs) %s' % name
        sys.stdout.flush()

        choices = EstimatorClass.param_choices()

        for i, tag in enumerate(tags):
            for j, params in enumerate(choices):
                key = 'intrinsicresults-intermediary-class={0}-tag={1}-i={2}-j={3}'.format(EstimatorClass, tag, i, j)
                # Start jobs for which we don't have a result already
                if key not in all_processed:
                    tasks.computeScoreJob_task.delay(name, EstimatorClass, params, tag, i, j, DATASETCHOICE, ERRORMETRIC, RESULTS_DIR, USE_L1, isFinalScore=False)
                else:
                    print 'Skipped (already processed): {0}'.format(key)


def aggregate_comparison_experiment(DATASETCHOICE, ALL_TAGS, ERRORMETRIC, USE_L1, RESULTS_DIR, ESTIMATORS):
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

    results = np.zeros((len(ESTIMATORS), ntags))
    best_choices = np.zeros((len(ESTIMATORS), ntags), np.int32)
    best_params = [[{} for x in range(ntags)] for x in range(len(ESTIMATORS))]

    print 'Waiting for all results to be computed...'
    nchoices_forclass = {}

    for e, (name, EstimatorClass) in enumerate(ESTIMATORS):
        choices = EstimatorClass.param_choices()
        nchoices = len(choices)

        nchoices_forclass[EstimatorClass] = nchoices

    resulthandler.wait_all_results(nchoices_forclass, ntags)

    print 'Collecting scores for all parameter configurations...'
    # Generate HTML using the results
    gen = html.Generator('Intrinsic image results', RESULTS_DIR)
    for e, (name, EstimatorClass) in enumerate(ESTIMATORS):
        print 'Evaluating %s' % name
        sys.stdout.flush()
        gen.heading(name)

        choices = EstimatorClass.param_choices()
        nchoices = len(choices)

        # Collect intermediary results from workers
        scores = resulthandler.gather_intermediary_results(EstimatorClass, ntags, nchoices)

        # Hold-one-out cross-validation
        print '  Final scores:'
        sys.stdout.flush()
        for i, tag in enumerate(tags):
            # Get the best parameter configuration
            other_inds = range(i) + range(i+1, ntags)
            total_scores = np.sum(scores[other_inds, :], axis=0)
            best_choice = np.argmin(total_scores)
            bestparam = choices[best_choice]

            score = scores[i, best_choice]
            results[e, i] = score

            gen.text('%s: %1.3f' % (tag, score))
            gen.text('Best parameters %s' % (bestparam))
            print '    %s: %1.3f' % (tag, score)

        print '    average: %1.3f' % np.mean(results[e, :])

        gen.divider()

    gen.heading('Mean error')
    ranks = [sp.stats.rankdata(results[:, i]) for i in range(ntags)]
    fullrankarr = np.transpose(np.vstack(ranks))

    for e, (name, EstimatorClass) in enumerate(ESTIMATORS):
        avg = np.mean(results[e, :])
        avgrank = np.mean(fullrankarr[e, :])
        gen.text('%s: mean error %1.3f; mean rank %1.2f' % (name, avg, avgrank))

    # Save valuable data to file
    with open(os.path.join(RESULTS_DIR, 'ALLDATA.dat'), 'w') as f:
        pickle.dump({'version': '1.0', 'results': results, 'best_params': best_params, 'date': datetime.datetime.now()}, f, protocol=2)


def computeScoreJob(name, EstimatorClass, params, tag, i, j, DATASETCHOICE, ERRORMETRIC, RESULTS_DIR, USE_L1, isFinalScore):
    """
    Input paramsDict:
        {'estimator' : (name, EstimatorClass)}
        {'params' : params with which the estimation will be called}
        {'tag' : the estimated object's (image) tag}
        {'score_coords' : the coordinates in the score matrix (i, j)}
    """

    # Estimators know what input they expect (grayscale image, color image, etc.)
    inp = EstimatorClass.get_input(tag, DATASETCHOICE)
    if isFinalScore:
        inp = inp + (USE_L1,)

    if ERRORMETRIC == 0:
        true_shading = intrinsic.load_object(tag, 'shading', DATASETCHOICE)
        true_refl = intrinsic.load_object(tag, 'reflectance', DATASETCHOICE)
        true_refl = np.mean(true_refl, axis=2)
        mask = intrinsic.load_object(tag, 'mask', DATASETCHOICE)
    elif ERRORMETRIC == 1:
        judgements = intrinsic.load_object(tag, 'judgements', DATASETCHOICE)
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

    if isFinalScore:
        key = 'intrinsicresults-final-class={0}-tag={1}-i={2}-j={3}'.format(EstimatorClass, tag, i, j)
        value = (score, est_shading, est_refl)
    else:
        key = 'intrinsicresults-intermediary-class={0}-tag={1}-i={2}-j={3}'.format(EstimatorClass, tag, i, j)
        value = score

    packed = packer.packb(value, version='1.0')

    return key, packed



