#!/bin/bash

# Begin configuration section
first_beam=10.0 # Beam used in initial, speaker-indep. pass
first_max_active=2000 # max-active used in initial pass.
alignment_model=
adapt_model=
final_model=
stage=0
acwt=0.083333 # Acoustic weight used in getting fMLLR transforms, and also in 
              # lattice generation.
max_active=7000
beam=13.0
lattice_beam=6.0
nj=4
silence_weight=0.01
cmd=run.pl
si_dir=
fmllr_update_type=full
num_threads=1 # if >1, will use gmm-latgen-faster-parallel
parallel_opts=  # If you supply num-threads, you should supply this too.
skip_scoring=false
scoring_opts=
kaldi=/kaldi-trunk/src

# End configuration section
echo "$0 $@"  # Print the command line for logging

[ -f ./path.sh ] && . ./path.sh; # source the path.
. parse_options.sh || exit 1;

if [ $# != 4 ]; then
   echo "Wrong #arguments ($#, expected 3)"
   echo "Usage: steps/decode_fmllr.sh [options] <graph-dir> <data-dir> <decode-dir>"
   echo " e.g.: steps/decode_fmllr.sh exp/tri2b/graph_tgpr data/test_dev93 exp/tri2b/decode_dev93_tgpr"
   echo "main options (for others, see top of script file)"
   echo "  --config <config-file>                   # config containing options"
   echo "  --nj <nj>                                # number of parallel jobs"
   echo "  --cmd <cmd>                              # Command to run in parallel with"
   echo "  --adapt-model <adapt-mdl>                # Model to compute transforms with"
   echo "  --alignment-model <ali-mdl>              # Model to get Gaussian-level alignments for"
   echo "                                           # 1st pass of transform computation."
   echo "  --final-model <finald-mdl>               # Model to finally decode with"
   echo "  --si-dir <speaker-indep-decoding-dir>    # use this to skip 1st pass of decoding"
   echo "                                           # Caution-- must be with same tree"
   echo "  --acwt <acoustic-weight>                 # default 0.08333 ... used to get posteriors"
   echo "  --num-threads <n>                        # number of threads to use, default 1."
   echo "  --parallel-opts <opts>                   # e.g. '-pe smp 4' if you supply --num-threads 4"
   echo "  --scoring-opts <opts>                    # options to local/score.sh"
   exit 1;
fi


graphdir=$1
data=$2
srcdir=$3
dir=`echo $4 | sed 's:/$::g'` # remove any trailing slash.

sdata=`dirname $data`
JOB=`basename $data`

echo "dir = $dir  JOB = $JOB"

thread_string=
[ $num_threads -gt 1 ] && thread_string="-parallel --num-threads=$num_threads"

## build up the dataset
#utils/utt2spk_to_spk2utt.pl $data/utt2spk > $data/spk2utt || exit 1;
#utils/filter_scp.pl $data/spk2utt $sdata/cmvn.scp > $data/cmvn.scp || exit 1;
#utils/filter_scp.pl $data/spk2utt $sdata/reco2file_and_channel > $data/reco2file_and_channel || exit 1;
#utils/filter_scp.pl $data/utt2spk $sdata/segments > $data/segments || exit 1;

splice_opts=`cat $srcdir/splice_opts 2>/dev/null` # frame-splicing options.
cmvn_opts=`cat $srcdir/cmvn_opts 2>/dev/null`

silphonelist=`cat $graphdir/phones/silence.csl` || exit 1;

# Some checks.  Note: we don't need $srcdir/tree but we expect
# it should exist, given the current structure of the scripts.
for f in $graphdir/HCLG.fst $data/feats.scp $srcdir/tree; do
  [ ! -f $f ] && echo "$0: no such file $f" && exit 1;
done

## Work out name of alignment model. ##
if [ -z "$alignment_model" ]; then
  if [ -f "$srcdir/final.alimdl" ]; then alignment_model=$srcdir/final.alimdl;
  else alignment_model=$srcdir/final.mdl; fi
fi
[ ! -f "$alignment_model" ] && echo "$0: no alignment model $alignment_model " && exit 1;
##

## Some checks, and setting of defaults for variables.
[ -z "$adapt_model" ] && adapt_model=$srcdir/final.mdl
[ -z "$final_model" ] && final_model=$srcdir/final.mdl
for f in $adapt_model $final_model; do
  [ ! -f $f ] && echo "$0: no such file $f" && exit 1;
done
##

## Set up the unadapted features "$sifeats"
if [ -f $srcdir/final.mat ]; then feat_type=lda; else feat_type=delta; fi
echo "$0: feature type is $feat_type";
case $feat_type in
  delta) sifeats="ark,s,cs:$kaldi/featbin/apply-cmvn $cmvn_opts --utt2spk=ark:$sdata/$JOB/utt2spk scp:$sdata/$JOB/cmvn.scp scp:$sdata/$JOB/feats.scp ark:- | $kaldi/featbin/add-deltas ark:- ark:- |";;
  lda) sifeats="ark,s,cs:$kaldi/featbin/apply-cmvn $cmvn_opts --utt2spk=ark:$sdata/$JOB/utt2spk scp:$sdata/$JOB/cmvn.scp scp:$sdata/$JOB/feats.scp ark:- | $kaldi/featbin/splice-feats $splice_opts ark:- ark:- | $kaldi/featbin/transform-feats $srcdir/final.mat ark:- ark:- |";;
  *) echo "Invalid feature type $feat_type" && exit 1;
esac
##

## Do the speaker-independent decoding, if --si-dir option not present. ##
if [ $stage -le 0 ]; then
  echo "$0: start si decoding"
  $kaldi/gmmbin/gmm-latgen-faster --max-active=$first_max_active --beam=$first_beam --lattice-beam=6.0 \
    --acoustic-scale=$acwt --allow-partial=true --word-symbol-table=$graphdir/words.txt \
    $alignment_model $graphdir/HCLG.fst "$sifeats" "ark:|gzip -c > $dir/lat.si.$JOB.gz" 2> $dir/log/lat.si.$JOB.log || exit 1;
fi
##

## Now get the first-pass fMLLR transforms.
if [ $stage -le 1 ]; then
  echo "$0: getting first-pass fMLLR transforms."
  gunzip -c $dir/lat.si.$JOB.gz | $kaldi/latbin/lattice-to-post --acoustic-scale=$acwt ark:- ark:- | \
    $kaldi/bin/weight-silence-post $silence_weight $silphonelist $alignment_model ark:- ark:- | \
    $kaldi/gmmbin/gmm-post-to-gpost $alignment_model "$sifeats" ark:- ark:- | \
    $kaldi/gmmbin/gmm-est-fmllr-gpost --fmllr-update-type=$fmllr_update_type \
    --spk2utt=ark:$sdata/$JOB/spk2utt $adapt_model "$sifeats" ark,s,cs:- \
    ark:$dir/pre_trans.$JOB 2> $dir/log/fmllr_pass1.$JOB.log || exit 1;
fi
##

pass1feats="$sifeats $kaldi/featbin/transform-feats --utt2spk=ark:$sdata/$JOB/utt2spk ark:$dir/pre_trans.$JOB ark:- ark:- |"

## Do the main lattice generation pass.  Note: we don't determinize the lattices at
## this stage, as we're going to use them in acoustic rescoring with the larger 
## model, and it's more correct to store the full state-level lattice for this purpose.
if [ $stage -le 2 ]; then
  echo "$0: doing main lattice generation phase"
  $kaldi/gmmbin/gmm-latgen-faster --max-active=$max_active --beam=$beam --lattice-beam=$lattice_beam \
    --acoustic-scale=$acwt --determinize-lattice=false \
    --allow-partial=true --word-symbol-table=$graphdir/words.txt \
    $adapt_model $graphdir/HCLG.fst "$pass1feats" "ark:|gzip -c > $dir/lat.tmp.$JOB.gz" \
    2> $dir/log/decode.$JOB.log || exit 1;
fi
##

## Do a second pass of estimating the transform-- this time with the lattices
## generated from the alignment model.  Compose the transforms to get
## $dir/trans.1, etc.
if [ $stage -le 3 ]; then
  echo "$0: estimating fMLLR transforms a second time."
  $kaldi/latbin/lattice-determinize-pruned --acoustic-scale=$acwt --beam=4.0 \
    "ark:gunzip -c $dir/lat.tmp.$JOB.gz|" ark:- | \
    $kaldi/latbin/lattice-to-post --acoustic-scale=$acwt ark:- ark:- | \
    $kaldi/bin/weight-silence-post $silence_weight $silphonelist $adapt_model ark:- ark:- | \
    $kaldi/gmmbin/gmm-est-fmllr --fmllr-update-type=$fmllr_update_type \
    --spk2utt=ark:$sdata/$JOB/spk2utt $adapt_model "$pass1feats" \
    ark,s,cs:- ark:$dir/trans_tmp.$JOB 2> $dir/log/fmllr_pass2.$JOB.log
  $kaldi/featbin/compose-transforms --b-is-affine=true ark:$dir/trans_tmp.$JOB ark:$dir/pre_trans.$JOB \
    ark:$dir/trans.$JOB  2> $dir/log/com_trans.$JOB.log || exit 1;
fi
##

feats="$sifeats $kaldi/featbin/transform-feats --utt2spk=ark:$sdata/$JOB/utt2spk ark:$dir/trans.$JOB ark:- ark:- |"

# Rescore the state-level lattices with the final adapted features, and the final model
# (which by default is $srcdir/final.mdl, but which may be specified on the command line,
# useful in case of discriminatively trained systems).
# At this point we prune and determinize the lattices and write them out, ready for 
# language model rescoring.

if [ $stage -le 4 ]; then
  echo "$0: doing a final pass of acoustic rescoring."
  $kaldi/gmmbin/gmm-rescore-lattice $final_model "ark:gunzip -c $dir/lat.tmp.$JOB.gz|" "$feats" ark:- | \
    $kaldi/latbin/lattice-determinize-pruned --acoustic-scale=$acwt --beam=$lattice_beam ark:- \
    "ark:|gzip -c > $dir/lat.$JOB.gz" 2> $dir/log/acoustic_rescore.$JOB.log || exit 1;
  rm $dir/lat.tmp.$JOB.gz
fi

LMWT=12
symtab=$graphdir/words.txt
word_ins_penalty=0.0
if [ $stage -le 5 ]; then
$kaldi/latbin/lattice-scale --inv-acoustic-scale=$LMWT "ark:gunzip -c $dir/lat.$JOB.gz|" ark:- | \
  $kaldi/latbin/lattice-add-penalty --word-ins-penalty=$word_ins_penalty ark:- ark:- | \
  $kaldi/latbin/lattice-best-path --word-symbol-table=$symtab \
    ark:- ark,t:$dir/tra.$JOB 2> $dir/log/best_path.$JOB.log || exit 1;
fi

#if [ $stage -le 6 ]; then
#$kaldi/latbin/lattice-align-words $graphdir/phones/word_boundary.int $final_model "ark:gunzip -c $dir/lat.$JOB.gz|" ark:- | \
#  $kaldi/latbin/lattice-to-ctm-conf --decode-mbr=true --acoustic-scale=0.08333333333  ark:- - | \
#  utils/int2sym.pl -f 5 $graphdir/words.txt | utils/convert_ctm.pl $sdata/$JOB/segments $sdata/$JOB/reco2file_and_channel > $dir/ctm.$JOB
#fi

#rm -rf $dir/lat.si.$JOB.gz $dir/pre_trans.$JOB $dir/lat.tmp.$JOB.gz $dir/trans_tmp.$JOB $dir/trans.$JOB $dir/lat.${JOB}.gz
# keep lat.1.gz - needed to produce subtitle files!
rm -rf $dir/lat.si.$JOB.gz $dir/pre_trans.$JOB $dir/lat.tmp.$JOB.gz $dir/trans_tmp.$JOB $dir/trans.$JOB

exit 0;
