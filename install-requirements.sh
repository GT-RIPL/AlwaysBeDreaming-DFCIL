while read requirement
    do conda install --yes $requirement || pip install $requirement
done < requirements.txt