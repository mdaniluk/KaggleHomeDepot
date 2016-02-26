package lucene;

import java.io.*;
import java.nio.file.Paths;

import java.util.Vector;


import com.opencsv.CSVReader;
import com.opencsv.CSVWriter;
import org.apache.lucene.analysis.Analyzer;
import org.apache.lucene.analysis.en.EnglishAnalyzer;
import org.apache.lucene.analysis.standard.StandardAnalyzer;
import org.apache.lucene.document.Document;
import org.apache.lucene.index.DirectoryReader;
import org.apache.lucene.index.IndexReader;
import org.apache.lucene.queryparser.classic.QueryParser;
import org.apache.lucene.search.IndexSearcher;
import org.apache.lucene.search.Query;
import org.apache.lucene.search.ScoreDoc;
import org.apache.lucene.search.TopDocs;
import org.apache.lucene.store.FSDirectory;

/**
 * Simple command-line based search demo.
 */
public class SearchFiles {

    private SearchFiles() {
    }

    private Vector<String> product_uid = new Vector<>();
    private Vector<String> id = new Vector<>();
    private Vector<String> query = new Vector<>();
    private String field = "contents";
    private Analyzer analyzer = new ThoMiHiDoAnalyzer(); //new EnglishAnalyzer(); //new StandardAnalyzer();
    private QueryParser parser = new QueryParser(field, analyzer);
    private IndexSearcher searcher;

    /**
     * Simple command-line based search demo.
     */
    public static void main(String[] args) throws Exception {

        String index = "data/index_standard";
        IndexReader reader = DirectoryReader.open(FSDirectory.open(Paths.get(index)));
        SearchFiles sf = new SearchFiles();
        sf.searcher = new IndexSearcher(reader);

        sf.calculateScores();
        reader.close();
    }

    public void readData(String filename) throws Exception {
        CSVReader reader = new CSVReader(new FileReader(filename));

        String[] dataRow;
        boolean isFirstLineRead = false;

        while ((dataRow = reader.readNext()) != null) {
            if (isFirstLineRead == false) {
                isFirstLineRead = true;
                continue;
            }
            id.add(dataRow[0]);
            product_uid.add(dataRow[1]);
            query.add(dataRow[3]);
        }
        // Close the file once all data has been read.
        reader.close();

        // End the printout with a blank line.
        System.out.println("reading data finished");
    }

    class Scores {
        public float luceneScore = (float) 0.0;
        public int rankingPlace = 21;
        public int inRanking = 0;
    }

    public Scores getScore(String q, String uid) throws Exception {
        q = q.trim();
        Query query = parser.parse(QueryParser.escape(q));
        int numTotalHits = 20;
        TopDocs results = searcher.search(query, numTotalHits);
        ScoreDoc[] hits = results.scoreDocs;

        Scores scores = new Scores();

        for (int i=0; i<hits.length; i++) {
            int fileIdx = hits[i].doc;
            Document doc = searcher.doc(fileIdx);
            File path = new File(doc.get("path"));
            String filename = path.getName();
            if (filename.equals(uid)) {
                scores.luceneScore = hits[i].score;
                scores.inRanking = 1;
                scores.rankingPlace = i;
                break;
            }
        }
        return scores;
    }

    public void calculateScores() throws Exception {
        String dataPath = "data/train.csv";
        readData(dataPath);
        String outputPath = "data/lucene_features.csv";
        CSVWriter writer = new CSVWriter(new FileWriter(outputPath));

        for (int i=0; i<id.size(); i++) {
            Scores scores = getScore(query.get(i), product_uid.get(i));
            String[] features = new String[4];
            features[0] = id.get(i);
            features[1] = Integer.toString(scores.inRanking);
            features[2] = Integer.toString(scores.rankingPlace);
            features[3] = Float.toString(scores.luceneScore);
            writer.writeNext(features);
        }
        writer.close();
    }
}
