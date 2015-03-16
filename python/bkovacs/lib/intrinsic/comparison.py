import os
import sys

import numpy as np
import scipy as sp
from collections import OrderedDict

from lib.intrinsic import html, intrinsic, tasks, resulthandler, pyzhao2012
from lib.utils.data import common, whdr
from lib.utils.misc import packer, strhelper
from lib.utils.misc.progressbaraux import progress_bar
from lib.utils.misc.mathhelper import nanrankdata

SCORE_FILENAME = 'ALLDATA.dat'


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


def run_experiment(DATASETCHOICE, ALL_TAGS, ERRORMETRIC, USE_L1, RESULTS_DIR, ESTIMATORS, ORACLEEACHIMAGE, IMAGESFORALLPARAMS):
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

            print 'Estimating shading and reflectance for ' + tag

            for j, params in progress_bar(enumerate(choices)):
                estimator = EstimatorClass(**params)
                est_shading, est_refl = estimator.estimate_shading_refl(*inp)
                q_ent, s_ent = pyzhao2012.compute_entropy(est_refl)

                if ERRORMETRIC == 0:
                    scores[i, j] = intrinsic.score_image(true_shading, true_refl, est_shading, est_refl, mask)
                elif ERRORMETRIC == 1:
                    scores[i, j] = whdr.compute_whdr(est_refl, judgements)
                else:
                    raise ValueError('Unknown error metric choice: {0}'.format(ERRORMETRIC))

                if IMAGESFORALLPARAMS:
                    gen.text('Score: %s: %1.3f' % (tag, scores[i, j]))
                    gen.text('Quadratic entropy: %1.3f, Shannon entropy: %1.3f' % (q_ent, s_ent))
                    gen.text('Parameters %s' % (params))
                    save_estimates(gen, image, est_shading, est_refl, mask)

                print 'Params: {0}'.format(params)
                print 'Score: {0}'.format(scores[i, j])
                print_dot(count, ntags * nchoices)
                count += 1

        # Hold-one-out cross-validation
        print '  Final scores:'
        sys.stdout.flush()
        for i, tag in progress_bar(enumerate(tags)):
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

            if ORACLEEACHIMAGE:
                best_choice = np.argmin(scores[i, :])
            else:
                other_inds = range(i) + range(i+1, ntags)
                total_scores = np.sum(scores[other_inds, :], axis=0)
                best_choice = np.argmin(total_scores)

            bestparam = choices[best_choice]
            estimator = EstimatorClass(**bestparam)
            est_shading, est_refl = estimator.estimate_shading_refl(*inp)

            if ERRORMETRIC == 0:
                score = intrinsic.score_image(true_shading, true_refl, est_shading, est_refl, mask)
            elif ERRORMETRIC == 1:
                score = whdr.compute_whdr(est_refl, judgements)
            else:
                raise ValueError('Unknown error metric choice: {0}'.format(ERRORMETRIC))

            gen.text('%s: %1.3f' % (tag, score))
            gen.text('Best parameters %s' % (bestparam))
            save_estimates(gen, image, est_shading, est_refl, mask)

            print '    %s: %1.3f' % (tag, score)

            results[e, i] = score
        print '    average: %1.3f' % np.mean(results[e, :])

        gen.divider()

    gen.heading('Mean error')
    for e, (name, EstimatorClass) in enumerate(ESTIMATORS):
        avg = np.mean(results[e, :])
        gen.text('%s: %1.3f' % (name, avg))


'''
Comparison experiment
'''
def dispatch_comparison_experiment(DATASETCHOICE, ALL_TAGS, ERRORMETRIC, USE_L1, RESULTS_DIR, ESTIMATORS, RERUNALLTASKS):
    tags = ALL_TAGS
    if not RERUNALLTASKS:
        print 'Getting all processed tasks, these won\'t be started again'
        all_processed = resulthandler.get_all_processed()

    if RERUNALLTASKS:
        keys_to_delete = []
        for e, (name, EstimatorClass) in enumerate(ESTIMATORS):
            choices = EstimatorClass.param_choices()

            for i, tag in enumerate(tags):
                for j, params in enumerate(choices):
                    jobid_params = OrderedDict({'i': i, 'j': j})
                    key = resulthandler.get_task_key('intermediary', EstimatorClass, tag, jobid_params)
                    keys_to_delete.append(key)

        print 'Deleting {0} results before starting new jobs'.format(len(keys_to_delete))
        resulthandler.delete_results(keys_to_delete)

    started_job_count = 0
    for e, (name, EstimatorClass) in enumerate(ESTIMATORS):
        print 'Evaluating (starting jobs) %s' % name
        sys.stdout.flush()

        choices = EstimatorClass.param_choices()

        for i, tag in enumerate(tags):
            for j, params in enumerate(choices):
                jobid_params = OrderedDict({'i': i, 'j': j})
                runtask = False
                if RERUNALLTASKS:
                    runtask = True
                else:
                    # Start jobs for which we don't have a result already
                    key = resulthandler.get_task_key('intermediary', EstimatorClass, tag, jobid_params)
                    runtask = key not in all_processed

                if runtask:
                    tasks.computeScoreJob_task.delay(EstimatorClass, params, tag, jobid_params, DATASETCHOICE, ERRORMETRIC, RESULTS_DIR, USE_L1, imageAsResult=False)
                    started_job_count += 1
                else:
                    print 'Skipped (already processed): {0}'.format(key)

    print 'Started {0} jobs'.format(started_job_count)


def aggregate_comparison_experiment(DATASETCHOICE, ALL_TAGS, ERRORMETRIC, USE_L1, RESULTS_DIR, ESTIMATORS, USESAVEDSCORES, ORACLEEACHIMAGE, PARTIALRESULTS):
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

    ioerr = False
    if USESAVEDSCORES:
        try:
            dic = packer.funpackb_version(1.1, os.path.join(RESULTS_DIR, SCORE_FILENAME))
            allscores = dic['allscores']
        except IOError:
            print 'IOError happened when reading scores file, falling back to using Redis server to retrieve data...'
            ioerr = True

    if ioerr:
        USESAVEDSCORES = False

    if PARTIALRESULTS:
        allscores = {}
    elif not USESAVEDSCORES:
        print 'Waiting for all results to be computed...'
        nchoices_forclass = {}
        for e, (name, EstimatorClass) in enumerate(ESTIMATORS):
            choices = EstimatorClass.param_choices()
            nchoices = len(choices)

            nchoices_forclass[EstimatorClass] = nchoices

        resulthandler.wait_all_results(nchoices_forclass, ntags)
        allscores = {}

    print 'Collecting scores for all parameter configurations...'
    # Generate HTML using the results
    gen = html.Generator('Intrinsic image results', RESULTS_DIR)
    for e, (name, EstimatorClass) in enumerate(ESTIMATORS):
        print 'Evaluating %s' % name
        sys.stdout.flush()
        gen.heading(name)

        choices = EstimatorClass.param_choices()
        nchoices = len(choices)

        if USESAVEDSCORES:
            scores = allscores[str(EstimatorClass)]
        else:
            # Collect intermediary results from workers
            scores = resulthandler.gather_intermediary_results(EstimatorClass, ntags, nchoices, PARTIALRESULTS)
            allscores[str(EstimatorClass)] = scores

        # Hold-one-out cross-validation
        print '  Final scores:'
        sys.stdout.flush()
        for i, tag in enumerate(tags):
            # Get the best parameter configuration
            try:
                if ORACLEEACHIMAGE:
                    best_choice = np.nanargmin(scores[i, :])
                else:
                    other_inds = range(i) + range(i+1, ntags)
                    total_scores = np.sum(scores[other_inds, :], axis=0)
                    best_choice = np.nanargmin(total_scores)
            except ValueError:
                # nanargmin found an all-NaN column, i.e. we don't have any results for a specific tag
                best_choice = 0

            bestparam = choices[best_choice]
            score = scores[i, best_choice]
            print 'bestparam: {0}, score: {1}'.format(bestparam, score)
            results[e, i] = score

            gen.text('%s: %1.3f' % (tag, score))
            gen.text('Best parameters %s' % (bestparam))
            print '    %s: %1.3f' % (tag, score)

        print '    average: %1.3f' % np.nanmean(results[e, :])

        gen.divider()

    gen.heading('Mean error')
    ranks = [nanrankdata(results[:, i]) for i in range(ntags)]
    fullrankarr = np.transpose(np.vstack(ranks))

    for e, (name, EstimatorClass) in enumerate(ESTIMATORS):
        avg = np.nanmean(results[e, :])
        avgrank = np.nanmean(fullrankarr[e, :])
        gen.text('%s: mean error %1.3f; mean rank %1.2f' % (name, avg, avgrank))

    # Save valuable data to file
    packer.fpackb({'results': results, 'allscores': allscores, 'best_params': best_params, 'tags': tags}, 1.1, os.path.join(RESULTS_DIR, SCORE_FILENAME))


'''
Compute certain predefined jobs
'''
def dispatch_predefined_jobs(job_params, DATASETCHOICE, ERRORMETRIC, USE_L1, RESULTS_DIR, RERUNALLTASKS, processed_tasks_filepath):
    if RERUNALLTASKS:
        all_processed = []
    else:
        print 'Getting all processed tasks, these won\'t be started again'
        all_processed = resulthandler.get_all_processed_from_file(processed_tasks_filepath)

    started_job_count = 0
    for job_param in progress_bar(job_params):
        # Start jobs for which we don't have a result already
        rest_job_param = OrderedDict({'classnum': job_param['classnum'], 'samplenum': job_param['samplenum']})
        key = resulthandler.get_task_key('imageAsResult', job_param['EstimatorClass'], job_param['tag'], rest_job_param)
        #runtask = key not in all_processed
        runtask = not os.path.exists(os.path.join(RESULTS_DIR, job_param['tag'], '{0}-classnum{1}-samplenum{2}-refl.png'.format(job_param['tag'], job_param['classnum'], job_param['samplenum']))) or not os.path.exists(os.path.join(RESULTS_DIR, job_param['tag'], '{0}-classnum{1}-samplenum{2}-shading.png'.format(job_param['tag'], job_param['classnum'], job_param['samplenum'])))

        if runtask:
            tasks.computeScoreJob_task.delay(job_param['EstimatorClass'], job_param['params'], job_param['tag'], rest_job_param, DATASETCHOICE, ERRORMETRIC, RESULTS_DIR, USE_L1, imageAsResult=True)
            started_job_count += 1
        else:
            print 'Skipped (already processed): {0}'.format(key)

    print 'Started {0} jobs'.format(started_job_count)


def aggregate_predifined_jobs(job_params, RESULTS_DIR, processed_tasks_filepath):
    resulthandler.gather_all_jobresults(job_params, RESULTS_DIR, processed_tasks_filepath)


def computeScoreJob(EstimatorClass, params, tag, jobid_params, DATASETCHOICE, ERRORMETRIC, RESULTS_DIR, USE_L1, imageAsResult):
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

    estimator = EstimatorClass(**params)
    try:
        est_shading, est_refl = estimator.estimate_shading_refl(*inp)
        if ERRORMETRIC == 0:
            score = intrinsic.score_image(true_shading, true_refl, est_shading, est_refl, mask)
        elif ERRORMETRIC == 1:
            score = whdr.compute_whdr(est_refl, judgements)
        else:
            raise ValueError('Unknown error metric choice: {0}'.format(ERRORMETRIC))
    except sp.linalg.LinAlgError:
        score = np.nan

    if imageAsResult:
        key = resulthandler.get_task_key('imageAsResult', EstimatorClass, tag, jobid_params)
        value = (score, est_shading, est_refl)
    else:
        key = resulthandler.get_task_key('intermediary', EstimatorClass, tag, jobid_params)
        value = score

    packed = packer.packb(value, version='1.0')

    return key, packed

