package ch04_.statement.sec04.forloop;

public class PrintFrom1To10Example3 {
	public static void main(String[] args) {
		
		// 조건식을 기술하지 않으면 무한루프
		for(int i=1; 1; ; ) { // 1,1,1,1
			System.out.println(i + " "); 
		}
	}
}

public class PrintFrom1To10Example3 {
    public static void main(String[] args) {
        
        // 조건식을 기술하지 않으면 무한루프
        for(int i=1; ; ) { // Removed the second semicolon after the first 'for' expression
            System.out.println(i + " ");
        }
    }
}
