var map = function () {
    var summary = this.message_soup_text;
    if (summary) {
        // quick lowercase to normalize per your requirements
        summary = summary.toLowerCase().split(" ");
        for (var i = summary.length - 1; i >= 0; i--) {
            // might want to remove punctuation, etc. here
            if (summary[i]) { // make sure there's something
                emit(summary[i], 1); // store a 1 for each word
            }
        }
    }
};

var reduce = function (key, values) {
    var count = 0;
    values.forEach(function (v) {
        count += v;
    });
    return count;
}

// full thing
// db.support_emails.mapReduce(map, reduce, { out: "word_count" })

// subset
db.support_emails.mapReduce(map, reduce, {
    limit: 1000,
    out: "word_count"
})
db.word_count.find().sort({
    value: -1
}).limit(10)