package ch05_.references.sec07.multidimension;

public class MultidimensionalArrayByValueListExample {
	public static void main(String[] args) {
		//2차원 배열 생성(행과 열의 구조가 동일할 필요는 없음)
		int[][] scores = {
				{ 80, 90, 96 }, // 0행 : 3열
				{ 76, 88 } // 1행 : 2열
		};

		// 문제 : 각 반의 총점과 평균을 구하라
		int totalStudent = 0;
		int totalSum = 0;
		for(int i=0; i<scores.length; i++) { //반의 수만큼 반복, 0, 1
			int totalClass =0;							// 반별 총점(추가)
			totalStudent += scores[i].length; 			//반의 학생 수 합산
			for(int k=0; k<scores[i].length; k++) { 	//해당 반의 학생 수만큼 반복
				totalSum += scores[i][k]; 				//학생 점수 합산
				totalClass += scores[i][k];
			}
			System.out.printf("학급별(%d) : 총점(%d), 평균(%f)\n ", i, totalClass, (float)totalClass / scores[i].length);
		}
		
		double totalAvg = (double) totalSum / totalStudent;
		System.out.println("전체 학생의 평균 점수: " + totalAvg);
	}
}