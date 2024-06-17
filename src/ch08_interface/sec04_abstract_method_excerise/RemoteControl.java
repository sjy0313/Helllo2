package ch08_interface.sec04_abstract_method_excerise;
/*
 * 추상메소드 : public abstract
 * 
 */
public interface RemoteControl {
	//상수 필드
	// 타입 상수명 = 값;
	int MAX_VOLUME = 10;
	int MIN_VOLUME = 0;

	//추상 메소드 : public method 암묵적으로 지정
	void turnOn();
	void turnOff();
	//void setVolume(int volume);
	public abstract void setVolume(int volume);
}
	