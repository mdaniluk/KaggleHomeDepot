package lucene;

import java.io.FileReader;

import org.apache.lucene.analysis.Analyzer;
import org.apache.lucene.analysis.TokenStream;
import org.apache.lucene.analysis.Tokenizer;
import org.apache.lucene.analysis.core.*;
import org.apache.lucene.analysis.en.EnglishPossessiveFilter;
import org.apache.lucene.analysis.en.PorterStemFilter;
import org.apache.lucene.analysis.standard.StandardAnalyzer;
import org.apache.lucene.analysis.standard.StandardFilter;
import org.apache.lucene.analysis.standard.StandardTokenizer;
import org.apache.lucene.analysis.synonym.SynonymFilter;
import org.apache.lucene.analysis.synonym.SynonymMap;
import org.apache.lucene.analysis.synonym.WordnetSynonymParser;
import org.apache.lucene.analysis.util.StopwordAnalyzerBase;

/**
 * {@link Analyzer} for English.
 */
public final class ThoMiHiDoAnalyzer extends StopwordAnalyzerBase {
    private SynonymMap synonymMap = null;

    public ThoMiHiDoAnalyzer() {
        super(StandardAnalyzer.STOP_WORDS_SET);
        try {
            FileReader reader = new FileReader("data/prolog/wn_s.pl");
            WordnetSynonymParser parser = new WordnetSynonymParser(true, true, new WhitespaceAnalyzer());
            parser.parse(reader);
            synonymMap = parser.build();
        } catch (Exception e) {
            System.out.println(e.getMessage());
            e.printStackTrace();
            System.exit(1);
        }
    }

    @Override
    protected TokenStreamComponents createComponents(String fieldName) {
        final Tokenizer source = new StandardTokenizer();
        TokenStream result = new StandardFilter(source);
        //result = new EnglishPossessiveFilter(result);
        result = new LowerCaseFilter(result);
        result = new StopFilter(result, stopwords);
        //result = new PorterStemFilter(result);
        //result = new SynonymFilter(result, this.synonymMap, true);
        return new TokenStreamComponents(source, result);
    }
}